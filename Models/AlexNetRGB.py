# -*- coding: utf-8 -*-
"""
@author:Freya Gray
CS3120 
Final Project
Implementaion of the AlexNet Model and trained on the Plant village data set

Sources: 
Dataset: PlantVillage
Hughes, D. P., &amp; Salathé, M. (n.d.). PlantVillage Dataset - Images of Healthy and Diseased plants. 
https://www.kaggle.com/abdallahalidev/plantvillage-dataset. 

Model: AlexNet
Krizhevsky, A., Sutskever, I., & Hinton, G.E. (2012). ImageNet classification with deep convolutional neural networks. 
Communications of the ACM, 60, 84 - 90.
"""
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from pathlib import Path

#Checking if gpu is available
print(tf.config.list_physical_devices('GPU'))

#Setting path for dataset folder
colorFolder = Path('plantvillagedataset/color')

#loading images from disk and splitting data. 80% for training 20% for testing. Resizes images to AlexNet Specs
trainColor = tf.keras.preprocessing.image_dataset_from_directory(
    colorFolder, labels='inferred', label_mode='categorical', batch_size = 32, image_size=(224,
    224), validation_split=(.2), subset=('training'),seed = 13)

#getting images for validation
validColor = tf.keras.preprocessing.image_dataset_from_directory(
    colorFolder, labels='inferred', label_mode='categorical',batch_size = 32, image_size=(224,
    224), validation_split=(.2), subset=('validation'),seed = 13)


#getting and printing class names
classNamesColor = trainColor.class_names
print(classNamesColor)

#Buffered Prefetching. Keeps images in memory, ensures memory is not the bottleneck
AUTOTUNE = tf.data.experimental.AUTOTUNE
trainColor = trainColor.cache().prefetch(buffer_size = AUTOTUNE)
validColor = validColor.cache().prefetch(buffer_size = AUTOTUNE)

#defining normalization layer to standardize pixel values between [0,1]
normalization = layers.experimental.preprocessing.Rescaling(scale=1./255)

#defining local response normalization to AlexNet specifications from paper
def lrn(x): return tf.nn.local_response_normalization(input = x, depth_radius = 2, alpha = .00002, beta = .75, bias = 1)
         
#AlexNet model 
model = Sequential()
#normalization layer
model.add(normalization)

#Convolution 1 layer
model.add(layers.Conv2D(filters = 96, kernel_size = (11,11),strides = (4,4), padding = 'same', activation = 'relu', input_shape = (224,224,3)))
#Local response normalization 1
model.add(layers.Lambda(lrn))
#max pooling 1 layer
model.add(layers.MaxPool2D(pool_size = (3,3), strides = (2,2), padding = 'valid'))

#Convolution 2 layer
model.add(layers.Conv2D(filters = 256, kernel_size = (5,5),strides = (1,1), padding = 'same', activation = 'relu'))
#Local response normalization 2
model.add(layers.Lambda(lrn))
#max pooling 2 layer
model.add(layers.MaxPool2D(pool_size = (3,3), strides = (2,2), padding = 'valid'))

#convolution 3
model.add(layers.Conv2D(filters = 384, kernel_size = (3,3),strides = (1,1), padding = 'same', activation = 'relu'))
#convolution 4
model.add(layers.Conv2D(filters = 384, kernel_size = (3,3),strides = (1,1), padding = 'same', activation = 'relu'))
#convolution 5
model.add(layers.Conv2D(filters = 256, kernel_size = (3,3),strides = (1,1), padding = 'same', activation = 'relu'))

model.add(layers.Flatten())

#Fully connected 1
model.add(layers.Dense(4096, activation ='relu'))
model.add(layers.Dropout(.5))

#fully connected 2
model.add(layers.Dense(4096, activation ='relu'))
model.add(layers.Dropout(.5))

#output
model.add(layers.Dense(38, activation = 'softmax'))

#compile model
# train the model usign SGD
opt = tf.keras.optimizers.SGD(learning_rate=0.001,nesterov = True, momentum = 0.9)
model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics=["accuracy"])

#number of epochs to train model 
epochs=10
#fitting model
trainedModel = model.fit(trainColor, validation_data=validColor, epochs=epochs)
#print summary of model architecture
model.summary()
#evaluate trained model
predictions = model.evaluate(validColor, verbose = 1)
#print loss and accuracy of model using validation set
print(predictions)

#save model
model.save('SavedModels/AlexNetModel')

#recording accuracy and loss for both training and validation
accuracy = trainedModel.history['accuracy']
validationAccuracy = trainedModel.history['val_accuracy']
loss = trainedModel.history['loss']
validationLoss = trainedModel.history['val_loss']
epochsRange = range(epochs)

#Plotting loss and accuracy 
plt.plot(epochsRange, accuracy, label='Training Accuracy')
plt.plot(epochsRange, validationAccuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy For AlexNet')
plt.show()

plt.plot(epochsRange, loss, label='Training Loss')
plt.plot(epochsRange, validationLoss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss For AlexNet')
plt.show()










