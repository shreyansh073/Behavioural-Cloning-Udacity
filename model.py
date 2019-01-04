import csv
import tensorflow as tf
import numpy as np
from scipy import ndimage

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Conv2D,Dropout,Cropping2D

#parse data from the csv file
lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#read images and steering angle
images = []
measurements = []
for line in lines:
    if(line[3]=='steering'):
        continue
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = 'data/IMG/' + filename
        image = ndimage.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

x_train = np.array(images)
y_train = np.array(measurements)

### Build the neural network architecture ###
model = Sequential()

#preprocess the data
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160,320,3)))   #normalize the data to remove bias
model.add(Cropping2D(cropping = ((70,25),(0,0))))       #Crop the image to avoid unwanted data

model.add(Conv2D(filters=24, kernel_size=(5,5),strides=(2,2),activation='relu'))
model.add(Conv2D(filters=36, kernel_size=(5,5),strides=(2,2),activation='relu'))
model.add(Conv2D(filters=48, kernel_size=(5,5),strides=(2,2),activation='relu'))

model.add(Conv2D(filters=64, kernel_size=(3,3),strides=(1,1),activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3),strides=(1,1),activation='relu'))

model.add(Flatten())

model.add(Dense(100))
model.add(Dropout(0.5))

model.add(Dense(50))
model.add(Dropout(0.5))

model.add(Dense(1))

### Compile and train the model ###
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=5)

#save the model
model.save('model.h5')
