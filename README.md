# Image Classification on Diseased Plants
This project compares the performance of three different Neural Networks when trained and tested on the PlantVillage dataset

## Description
This project implements three neural networks, AlexNet, InceptionV3 and ZFNet. AlexNet and ZFNet were implemented following their respective research papers using TensorFlow. InceptionV3 was implemented using TensorFlow's transfer learning models. All models were trained and tested on the PlantVillage Dataset. This project was developed for Dr. Feng Jiang's Machine Learning class at Metropolitan State University of Denver. This project was a way for me to learn about neural networks while using a dataset that I found interesting. 

## Getting Started

### Dependencies

* The PlantVillage dataset is required to run the models. [PlantVillage Dataset](https://www.kaggle.com/abdallahalidev/plantvillage-dataset)
* TensorFlow and MatPlotLib must be installed before running the models

### Installing

* All the Python files contain paths to the dataset as well as paths to save the models. All paths can be modified. 

### Executing program

* Run the python files. allModels.py will need saved models before it can be run

### List of folders and files
* AccuracyAndLossGraphs - pyplot graphs of accuracy and loss for models. Generated for 10,25 and 50 epochs.
* ModelArchitecture - images of each models summary/architecture
* Models - contains all Python files for each model. Files included are
    * AlexNetRGB.py - implementation and training of AlexNet. Saves model to SavedModels
    * allModels.py - imports saved models, runs prediction from saved model
    * InceptionV3RGB.py - transfer learning of InceptionV3. Saves model to SavedModels
    * ZFNetRGB.py - implementation and training of ZFNet. Saves model to SavedModels
* FinalProject_Gray_Freya.pdf - project report
* FreyaGrayFinalPresentation.pdf - slides used for video presentation


## Authors
Freya Gray
[Freya's GitHub](https://github.com/freyakgray)

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments
Dataset: PlantVillage
* Hughes, D. P., &amp; Salathé, M. (n.d.). *PlantVillage Dataset - Images of Healthy and Diseased plants*. [Link](https://www.kaggle.com/abdallahalidev/plantvillage-dataset.) 

Model: AlexNet
* Krizhevsky, A., Sutskever, I., & Hinton, G.E. (2012). *ImageNet classification with deep convolutional neural networks*. Communications of the ACM, 60, 84 - 90.

Model: InceptionV3
* Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, & Zbigniew Wojna. (2015). 
*Rethinking the Inception Architecture for Computer Vision*.

Model: ZFNet
* Matthew D Zeiler, & Rob Fergus. (2013). *Visualizing and Understanding Convolutional Networks*.

TensorFlow has great tutorials which helped me learn how to use TensorFlow and implement the models.

Thank you to Dr. Feng Jiang for being a great teacher and teaching the machine learning class.




