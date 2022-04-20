import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras import backend as K
from keras.regularizers import l2
from keras.models import Model
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def resnet_layer(inputs,num_filters=16,kernel_size=3,strides=1,activation='relu',batch_normalization=True,conv_first=True):
    
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v1(input_shape, depth, num_classes=10):
    
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,kernel_initializer='he_normal')(y)
    outputs = Activation('softmax')(outputs)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

############### Loading the dataset ###########################
num_classes = 10
training_set = sio.loadmat('./train_32x32.mat')
x_train = np.transpose(training_set["X"], (3, 0, 1, 2))
y_train = training_set["y"]
y_train[y_train == 10] = 0

testing_set = sio.loadmat('./test_32x32.mat')
x_test = np.transpose(testing_set["X"], (3, 0, 1, 2))
y_test = testing_set["y"]
y_test[y_test == 10] = 0

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert class vectors to binary class matrices.
y_train = np.eye(num_classes)[y_train.reshape((-1,))]
y_test = np.eye(num_classes)[y_test.reshape((-1,))]

input_shape = x_test.shape[1:]
################### loading the original model and embedding the noise extractor layer ###############
model_name = 'svhn'
depth = 20
org_model = resnet_v1(input_shape=input_shape, depth=depth, num_classes=num_classes)
org_model.load_weights(model_name+'.hdf5')
opt = keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)
org_model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

#input(org_model.summary())
#print('model\'s accuray: ', org_model.evaluate(x_test,y_test,verbose=0)[1])

############################################################################
logit_model = Model(org_model.input,org_model.get_layer('dense').output)
logit_model.compile(loss=keras.losses.mean_squared_error,
		optimizer=opt,
		metrics=['accuracy'])
#######################################################################
acts_tensors = []
number_of_neurons = 0
for layer in logit_model.layers:
	if layer.name.find('activ') != -1:
		neurons=1
		for d in layer.output.shape[1:]:
			neurons *= int(d)
		number_of_neurons += neurons
		acts_tensors.append(K.flatten(layer.output))

all_activations = K.concatenate(acts_tensors)
get_off_count = K.function([logit_model.input],[K.sum(K.cast(K.equal(all_activations,0.0),K.floatx()))])

##################################################################
lambda_hyperparameter = 0.20

edgepoint_1st_loss_term = 0
for label in range(num_classes):
	edgepoint_1st_loss_term += keras.losses.mean_squared_error(K.max(logit_model.output),logit_model.output[:,label])

edgepoint_2nd_loss_term = 0
for i,layer in enumerate(logit_model.layers):
	if (layer.name.find('dense') != -1):
		if layer.units != num_classes:
			edgepoint_2nd_loss_term += K.mean((K.relu((-1.0)*K.flatten(layer.output))))

	if layer.name.find('conv') != -1:
		edgepoint_2nd_loss_term += K.mean((K.relu((-1.0)*K.flatten(layer.output))))

get_pert_grads = K.function([logit_model.input],K.gradients(edgepoint_1st_loss_term+lambda_hyperparameter*edgepoint_2nd_loss_term, logit_model.input))
get_2nd_loss_term_value = K.function([logit_model.input],[edgepoint_2nd_loss_term])
get_1st_loss_term_value = K.function([logit_model.input],[edgepoint_1st_loss_term])
get_total_loss = K.function([logit_model.input],[edgepoint_1st_loss_term+lambda_hyperparameter*edgepoint_2nd_loss_term])
#################################################################################################
input_img = np.random.random(size=x_train.shape[1:])

epochs = 15000
start_lr = 0.0001
best_loss = 100000
best_input_img = None
def schedule_lr(epoch):	
    if epoch < 5000:
        return start_lr
    elif epoch < 10000:
        return start_lr / 2.0
    else:
        return start_lr / 4.0

def plot_metrics(opt_info, file_name):
    plt.cla()
    plt.figure(figsize=(10,3*len(opt_info.keys())))
    colors = ['r','b','darkgreen','black']
    for i, metric in enumerate(opt_info.keys()):
        ax = plt.subplot(len(opt_info.keys()),1,1+i)
        y = opt_info[metric]
        x = range(len(y))
        ax.plot(x,y,color=colors[i], label=metric)
        ax.set_xlabel('round')
        ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(file_name+".png")
    plt.close()


opt_info = {'total loss':[],'first loss term':[], 'second loss term':[], 'off neuron':[] }

for e in range(epochs):
    lr = schedule_lr(e)	
    pert_grads = (get_pert_grads([input_img.reshape((1,)+input_img.shape)])[0]).reshape(input_img.shape)
    input_img = input_img - lr * pert_grads
    input_img = np.clip(input_img,0.0,1.0)

    first_term_value = get_1st_loss_term_value([input_img.reshape((1,)+input_img.shape)])[0]
    second_term_value = get_2nd_loss_term_value([input_img.reshape((1,)+input_img.shape)])[0]
    total_loss = get_total_loss([input_img.reshape((1,)+input_img.shape)])[0]
    off_count = get_off_count([input_img.reshape((1,)+input_img.shape)])[0]

    if total_loss <= best_loss:
        best_loss = total_loss
        best_input_img = input_img

    print('[',e,'] '," total loss: ", '{:0.5f}'.format(total_loss) , " first loss term: ", '{:0.5f}'.format(first_term_value),
    " second loss term: ", '{:0.5f}'.format(second_term_value),
        " Ratio of off neurons : ", '{:0.5f}'.format(off_count/number_of_neurons) ,  end='\r')

    opt_info['total loss'].append(total_loss)
    opt_info['first loss term'].append(first_term_value)
    opt_info['second loss term'].append(second_term_value)
    opt_info['off neuron'].append(off_count/number_of_neurons)

    if e%50 == 0:
        plot_metrics(opt_info=opt_info,file_name="./images/"+str(e))
        plot_metrics(opt_info=opt_info,file_name="./optimization-summary")
    

prediction_probabilities = (org_model.predict(best_input_img.reshape((1,)+best_input_img.shape))[0])
print("\n prediction probabilities : " , prediction_probabilities)
			







