# Description of the contents in this directory

Here is a brief description of what is done inside of each Python notebook file developed:

- **01_EDA.ipynb**: quick exploratory analysis of the dataset providing information about the patients.
- **02_image_tagging.ipynb**: labelling the images and distributing them into directories depending on the disease. Train/test split is also done. 
- **03_image_flipping.ipynb**: image vertically flipping treatment is done to make right images similar to the left ones. 
- **04_simple_CNN.ipynb**: training a simple CNN model for the entire dataset (8 classes).
- **05_VGG16.ipynb**: training a model using pretrained VGG16 for the entire dataset (8 classes).
- **06_ResNet.ipynb**: training a model using pretrained ResNet50 for the entire dataset (8 classes).
- **07_Resnet_7_classes.ipynb**: training a model using pretrained ResNet50 for the entire dataset except class O (7 classes).
- **08_Resnet_Retrained.ipynb**: retraining the model trained in notebook 07_Resnet_7_classes.ipynb using ResNet50 for the entire dataset except class O (7 classes).
- **09_Resnet_binary_Diabetes_Normal.ipynb**: training a model using pretrained ResNet50 for D and N classes (binary classification).
- **10_Resnet_flip.ipynb**: training a model using pretrained ResNet50 for the entire dataset except class O (7 classes). Images flipped vertically.
- **11_VGG_flip.ipynb**: training a model using pretrained ResNet50 for the entire dataset except class O (7 classes). Images flipped vertically.
- **12_VGG_binary_Cataract_Normal.ipynb**: training a model using pretrained VGG16 for C and N classes (binary classification). Images flipped vertically.

<br/>

**The models were trained on Google Colab. Label and treatment of the images were done locally to speed up the process.**
