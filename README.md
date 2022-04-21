# Ocular Disease Detection - Image Recognition

The purpose of this project is to detect ocular diseases using image recognition. The objective is to create a Deep Learning model that can identify between different diseases in fundus eye images.

**Ocular Disease Intelligent Recognition (ODIR)** dataset was downloaded from Kaggle and is a structured ophthalmic database of 5,000 patients with age, color fundus photographs from left and right eyes and doctors' diagnostic keywords from doctors, collected by Shanggong Medical Technology Co., Ltd. from different hospitals/medical centers in China.

Images are classified in eight labels including:

* Normal (N),
* Diabetes (D),
* Glaucoma (G),
* Cataract (C),
* Age related Macular Degeneration (A),
* Hypertension (H),
* Pathological Myopia (M),
* Other diseases/abnormalities (O)

Kaggle link to the dataset: https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k

According to 2010 World Health Organization data: There are more than 39 million blind people where 80% of them could have been prevented! This lack of prevention is especially true in developing countries where cataract is still the highest with 51% globally.

The current standard for the classification of diseases based on fundus photography, includes a manual estimation of injury locations and an analysis of their severity, which requires a lot of time by the ophthalmologist, also incurring high costs in the healthcare system. Therefore, it would be helpful to have automated methods for performing the analysis.

In the next image, a plot of the diagnosis distribution from the labelled images that appear in the ODIR dataset is shown:

![patologies.PNG](attachment:patologies.PNG)

## Deep Learning

In order to classify the images into the class to which they belong, a Deep Learning model build from convolutional neural networks (CNN) to extract features from the images followed by classification layers have been used. In this sense, two approaches have been carried out. The first one was to create a simple model made from scratch and the second one was to use pre-trained models that are more robust since they will be able to better detect the characteristics that define the images. In this second case, the ResNet50 and VGG16 models were tested.


### ResNet50

ResNet50 is a variant of ResNet model which has 48 Convolution layers along with 1 MaxPool and 1 Average Pool layer. It has 3.8 x 10^9 floating points operations. It is a widely used ResNet model and its architecture was explored in depth.

* This architecture can be used on computer vision tasks such as image classification, object localization or object detection.




![The-architecture-of-ResNet-50-model.png](attachment:The-architecture-of-ResNet-50-model.png)

### VGG16


VGG16 is a variant of VGG model with 16 convolution layers and also its architecture was studied in depth.

VGGNet-16 consists of 16 convolutional layers and is very appealing because of its very uniform Architecture. Similar to AlexNet, it has only 3x3 convolutions, but lots of filters (more computational time and parameters). It is currently the most preferred choice in the community for extracting features from images. The weight configuration of the VGGNet is publicly available and has been used in many other applications and challenges as a baseline feature extractor.

However, VGGNet consists in a high number of parameters, which can be a bit challenging to handle. VGG can be achieved through transfer Learning in which the model is pretrained on a dataset and the parameters are updated for better accuracy and you can use the parameters values.



![1_3-TqqkRQ4rWLOMX-gvkYwA.png](attachment:1_3-TqqkRQ4rWLOMX-gvkYwA.png)

# Project phases

## 0. Preprocessing. Train, validation, and test split.

Images from the ODIR dataset comes all in one directory and can be labelled using a csv file attached to it. The first step was to explore this csv file and come up with the correct way to label the images propertly.

Train and test was done in a 90% and 10% respectively distribution since some of the classes come with very few examples and less than 90 percent would be insufficient to train. From the 90 percent, 80% were selected for training and 20% for validation. Also the split was done selecting randomly the image files and respecting classes distribution.

*Notebooks: 01_EDA.ipynb, 02_image_tagging.ipynb*


## 1. Classification using all classes

At first, a model was build using all the classes with the idea to classify each image into one of the eight classes.

The limitation found  was that having so many categories of images to classify, the model was inaccurate and did not learn correctly making poor predictions. Simple architecture CNN, ResNet50 pretrained model and Vgg16 pretrained model were tested.

*Notebooks: 04_simple_CNN.ipynb, 05_VGG16.ipynb, 06_Resnet.ipynb*


## 2. Classification using all classes except 'Other (O)'

As the Other diseases/abnormalities (O) class contains many different observations from the images (keywords such as vitreous degeneration, drusen, epiretinal membrane, refractive media opacity...) it was concluded that this could cause problems for the model as there is not a clear pattern in the images that define that category. After making this observation decision to eliminate this class was taken. 

Resnet50 pretrained model was tested again without Other(O) class to see if any improvement was found. Also, this model was trained two times (second time using the result from the first time) in order to expand the training time and wait for improvements. No major improvements were found. 

*Notebooks: 07_Resnet_7_classes.ipynb, 08_Resnet_Retrained.ipynb*

## 3. Binary classification between Diabetes and Normal

In order to reduce the complexity of the problem, reducing the dimension of the problem was tried by choosing the categories Normal(N) and Diabetes(D). Diabetes and Normal classes were chosen for the binary classification because some of the predictions made before by the previous models tested tends to missclasify the images in these two classes. Resnet pretrained model was tested and poor results were achieve on the test set.

*Notebooks: 09_Resnet_binary_Diabetes_Normal.ipynb*


## 4. Flipping images

Initially, the model had left and right eye images as input. In this way, the images were not exactly similar since the pupil of the eye points into different directions depending if it is a right or left image.

That is why it was decided to flip vertically the right images so that they would all point to the same direction. With this image treatment, helping the model in the training process was tried with the objective to achieve a better predictive capacity.

Pretrained models Resnet50 and VGG16 were again tested using flipped images as inputs and also using all the classes except 'O'. Both models still performing poorly but VGG one classify images from some of the classes propertly, with quite good results distinguishing Cataract (C) class.

*Notebooks: 03_image_flipping.ipynb, 10_Resnet_flip.ipynb, 11_VGG_flip.ipynb*


## 5. Binary classification between Cataract and Normal

Since the number of categories to be classified is very high and the models tested had difficulties to learn the patterns from the classes, simplifying the problem was tried by selecting one of the most frequent ocular disesases, which is cataracts.

A model using pretrained model VGG16 with binary classification was tested using Cataract and Normal classes. The results obtained by this model have been very positive, reaching an accuracy of nearby 100% in the test set.

*Notebooks: 12_VGG_binary_Cataract_Normal.ipynb*

The decision to choose Cataract class was also made because the results obtained in notebook 11_VGG_flip.ipynb, where the model made relatively good prections for this class. Also, visually inspecting the images there is a clear difference between ocular fundus images from a Normal eye and an eye with Cataract disease, as it can be seen in the next image:

![C_N_image.png](attachment:C_N_image.png)

**Note: all the ipynb notebooks were run on Google Colab.**
