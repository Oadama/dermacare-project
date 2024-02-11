# ISIC 2019 - Skin Lesion Analysis Towards Melanoma Detection

## Getting started
### Dependencies

* Python 3.6.10
* TensorFlow 2.10.0
* Keras 2.10.0
* Pandas
* Numpy
* OpenCV-Python 4.7.0.72
* Pillow 9.1.1
* fastapi 0.109.0

### Datasets
This dataset contains the training data for the ISIC 2019 challenge, note that it already includes data from previous years (2018 and 2017).

The dataset for ISIC 2019 contains 25,331 images available for the classification of dermoscopic images among nine different diagnostic categories:

Melanoma
Melanocytic nevus
Basal cell carcinoma
Actinic keratosis
Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)
Dermatofibroma
Vascular lesion
Squamous cell carcinoma
None of the above

### How to use this dataset
* Extract if need be dataset from archive
* You have the choice with original or cropped and augmented dataset
* It has been notcied that even with augmented dataset, the prediction was not significant better

### Training
* The model used is a DenseNet 121

### Testing Results
* The accuracy on a 8 class output is 77%
* The model is performing less with bening pictures
* A binary outputs for the prediction will give much more accuracy.
* !!! At the moment the position of the lesion neither the age has no impact on the prediction !!!
* Hereunder the api under to test the code :
* - https://dermacare-project-6k7hrgqg8bqkjeospazdf3.streamlit.app/
