
'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 5 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import cv2
import numpy as np
import math
import sys
import os
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import TensorBoard

bin_dir="/tmp"
TARGET_DIR = bin_dir
CWD1 = os.getcwd()
print("CWD1: ", CWD1)
#!rm ngrok*

#os.chdir('..')
#!rm log
# Download Ngrok executable and extract to current directory
!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip ngrok-stable-linux-amd64.zip

# Fire up the TensorBoard in the background
LOG_DIR = './log'
get_ipython().system_raw(
   'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(LOG_DIR)
)

# Run ngrok to tunnel TensorBoard port 6006 to the outside world
get_ipython().system_raw('./ngrok http 6006 &')

# Get the public URL where we can access the colab TensorBoard web page
! curl -s http://localhost:4040/api/tunnels | python3 -c \
    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"

#Callbacks that allow to visualize dynamic graphs of training and test metrics
#tb_callback = TensorBoard(log_dir='./log', histogram_freq=1,
#                        write_graph=True,
#                        write_grads=True,
#                       batch_size=batch_size,
#                      write_images=True)

def relabel(labels):
    for idx, item in enumerate(labels):
        if item % 2 == 0:
            labels[idx] = 0     # even number
        else:
            labels[idx] = 1     # odd number
    return labels
  
  
# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = relabel(y_train)
y_test = relabel(y_test)

batch_size = 128
num_classes = 2
epochs = 5

if K.image_data_format() == 'channels_first':
  x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
  x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
  input_shape = (1, img_rows, img_cols)
else:
  x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
  x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
  input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#Classify between even and odd numbers
#x_train = relabel(x_train)
#x_test = relabel(x_test)


# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Create layers of CNN: linear stack of layers
model = Sequential()


# Method add() to add layers
# Conv2D: This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs.
# 2 convolutional layers
model.add(Conv2D(32, kernel_size=(5, 5),
                activation='relu',
                input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))

# Two pooling layers, dawnsample by a factor of 2
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.40))
model.add(Flatten())
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.40))
model.add(Dense(num_classes, activation='softmax'))

## Auxiliary functions for precision and recall
def precision(y_true, y_pred):  
    """Precision metric.  
    Only computes a batch-wise average of precision. Computes the precision, a
    metric for multi-label classification of how many selected items are
    relevant.
    """ 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))  
    precision = true_positives / (predicted_positives + K.epsilon())  
    return precision

def recall(y_true, y_pred): 
    """Recall metric. 
    Only computes a batch-wise average of recall. Computes the recall, a metric
    for multi-label classification of how many relevant items are selected. 
    """ 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1))) 
    recall = true_positives / (possible_positives + K.epsilon())  
    return recall
  
model.compile(loss=keras.losses.categorical_crossentropy,
     optimizer= keras.optimizers.SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False),
     metrics=['accuracy'])
              #,precision,recall])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
          #callbacks = [tb_callback])

score1 = model.evaluate(x_train, y_train, verbose = 0)
score2 = model.evaluate(x_test, y_test, verbose=0)
json_string = model.to_json()
model.save('model.h5')
model.save_weights('weights.h5')
#model.save_weights('weights.h5')
## Training loss and accuracy
#print('Train loss:', score1[0])
#print('Train accuracy:', score1[1])
## Test loss and accuracy
print('Test loss:', score2[0])
print('Test accuracy:', score2[1])
    
#json_string = model.to_json()
#model.save('model.h5')
#model.save_weights('weights.h5')


    
