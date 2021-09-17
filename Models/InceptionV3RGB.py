# -*- coding: utf-8 -*-
"""
@author:Freya Gray
CS3120 
Final Project
Transfer learning with Inception V3 Model on the Plant village data set
Sources: 
Dataset: PlantVillage
Hughes, D. P., &amp; Salathé, M. (n.d.). PlantVillage Dataset - Images of Healthy and Diseased plants. 
https://www.kaggle.com/abdallahalidev/plantvillage-dataset. 

Model: InceptionV3
Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, & Zbigniew Wojna. (2015). 
Rethinking the Inception Architecture for Computer Vision.
"""
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from pathlib import Path

#Checking if gpu is available
print(tf.config.list_physical_devices('GPU'))

#Setting path for correct folder
colorFolder = Path('plantvillagedataset/color')

#loading images from disk and splitting data. 80% for training 20% for testing. Resizes images to AlexNet Specs
trainColor = tf.keras.preprocessing.image_dataset_from_directory(
    colorFolder, labels='inferred', label_mode='categorical', batch_size = 32, image_size=(224,
    224), validation_split=(.2), subset=('training'),seed = 13)

#getting images for validation
validColor = tf.keras.preprocessing.image_dataset_from_directory(
    colorFolder, labels='inferred', label_mode='categorical',batch_size = 32, image_size=(224,
    224), validation_split=(.2), subset=('validation'),seed = 13)

#Getting image labels
classNamesColor = trainColor.class_names
print(classNamesColor)

#Buffered Prefetching. Keeps images in memory
AUTOTUNE = tf.data.experimental.AUTOTUNE
trainColor = trainColor.cache().prefetch(buffer_size = AUTOTUNE)
validColor = validColor.cache().prefetch(buffer_size = AUTOTUNE)

#Function that normalizes pixel values between [-1,1]
preproccess = tf.keras.applications.inception_v3.preprocess_input

#Inception V3 base model import
preTrainModel = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet',input_shape=(224,224,3))

#Freezing trainable weights
preTrainModel.trainable = False

#adding custom layers
input = tf.keras.Input(shape =(224,224,3))
#normalize pixel values
x = preproccess(input)
#get pretrained model with trainable layers frozen
x = preTrainModel(x, training = False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(.2)(x)
output = layers.Dense(38, activation = 'softmax')(x)
model = Model(input, output)

#compile model
# train the model usign SGD
opt = tf.keras.optimizers.SGD(learning_rate=0.001,nesterov = True,momentum = 0.9)
model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics=["accuracy"])
model.summary()

# number of epochs to train model
epochs=10
#fitting model
trainedModel = model.fit(trainColor, validation_data=validColor, epochs=epochs)
#print summary of model architecture
model.summary()

#evaluate model using validation set
predictions = model.evaluate(validColor, verbose = 1)
#print loss and accuracy of model using validation set
print(predictions)

#saving model
model.save('SavedModels/InceptionV3Model')

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
plt.title('Training and Validation Accuracy For InceptionV3')
plt.show()

plt.plot(epochsRange, loss, label='Training Loss')
plt.plot(epochsRange, validationLoss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss For InceptionV3')
plt.show()







