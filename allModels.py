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

Model: AlexNet
Krizhevsky, A., Sutskever, I., & Hinton, G.E. (2012). ImageNet classification with deep convolutional neural networks. 
Communications of the ACM, 60, 84 - 90.

Model: InceptionV3
Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, & Zbigniew Wojna. (2015). 
Rethinking the Inception Architecture for Computer Vision.

Model: ZFNet
Matthew D Zeiler, & Rob Fergus. (2013). Visualizing and Understanding Convolutional Networks.
"""

import tensorflow as tf
from pathlib import Path

#Setting path for correct folder
colorFolder = Path('plantvillagedataset/color')

#getting images for validation
validColor = tf.keras.preprocessing.image_dataset_from_directory(
    colorFolder, labels='inferred', label_mode='categorical',batch_size = 32, image_size=(224,
    224), validation_split=(.2), subset=('validation'),seed = 13)

#Getting image labels
classNamesColor = validColor.class_names

#Buffered Prefetching. Keeps images in memory, ensures memory is not the bottleneck
AUTOTUNE = tf.data.experimental.AUTOTUNE
validColor = validColor.cache().prefetch(buffer_size = AUTOTUNE)

#models. Uncomment to test a model. Number represents # of epochs model was trained on
#model = 'AlexNetModel50'
#model = 'InceptionV3Model50'
model = 'ZFNetModel50'

#load model from savedModels folder
loadedModel = tf.keras.models.load_model('SavedModels/' + model)
#evaluate model on validation set
validationPredictions = loadedModel.evaluate(validColor, verbose = 1)
#print loss and accuracy of model using validation set 
print(model + " Loss and Accuracy on Validation Set")
print(validationPredictions)
 
    




