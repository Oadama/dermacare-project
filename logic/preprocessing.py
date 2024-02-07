from params import *
import tensorflow as tf
import os
from tensorflow import keras
import glob
import cv2
import numpy as np
import pandas as pd
import random
from shutil import copyfile
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import preprocess_input

def initialize_dataset_from_file(url_file_name,extract:bool=False,archive_format:str=None):
    """
    Load tensorflow dataset and return local data path
    """
    data_dir = tf.keras.utils.get_file(
                                   origin=url_file_name,
                                   extract=extract,
                                   archive_format  = archive_format)
    return os.path.split(data_dir)[0]

def get_split_image_data(parent_path, child_path, img_height:int=IMAGE_HEIGHT, img_width:int=IMAGE_WIDTH,batch_size:int=BATCH_SIZE ):
    """
    get from targeted tensor path and return all existing files
    """
    path=os.path.join(parent_path + f'/{child_path}')
    bloc = tf.keras.utils.image_dataset_from_directory(
    path,
    image_size=(img_height, img_width),
    label_mode="categorical",
    batch_size=batch_size)

    return bloc

def class_names(df):
    # return the class names
    return df.class_names

def data_augmentation(img_height:int,img_width:int, val_rotation:float=0.1, val_zoom:float=0.1):

    result = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal",
                        input_shape=(img_height,
                                    img_width,
                                    3)),
        keras.layers.RandomRotation(val_rotation),
        keras.layers.RandomZoom(val_zoom),
    ]
    )
    return result

def normalisation(data_source):

    normalization_layer = keras.layers.Rescaling(1./255)
    return data_source.map(lambda x, y: (normalization_layer(x), y))

def images_augmentation_class(folder_class, augmented_folder,factor_mult:int, file_suffix:str='jpg'):
    """
    treat the whole existing files into folder_class for a given class.
    expect images with suffix_file == jpg
    """
    if not folder_class.exists():
        print(f'\n❌ Folder {folder_class} not found')
        return None
    if not augmented_folder.exists():
        print(f'\n❌ Folder {augmented_folder} not found. Please create it before')
        return None
        """
        os.mkdir augmented_folder
        print(f'\n Folder {augmented_folder} created')
        """

    for file in glob.glob(f'{folder_class}/*.{file_suffix}'):
        # copy first the original file
        fdst=f'{augmented_folder}/{file}'
        copyfile(file, fdst)
        #load the image
        img = cv2.imread(file)
        write_path='/'.join(file.split('/')[2:]).rstrip('.jpg')

        for i in range(factor_mult):
            X_aug=image_transforme(img, i+1)
            #save the data
            cv2.imwrite(f'{augmented_folder}/{write_path}_{i}.jpg', X_aug)

def transformer_augmentation(min_rotation, max_rotation):

    # Rotation range
    rotation_range = random.randint(min_rotation,max_rotation)


    datagen = ImageDataGenerator(
        featurewise_center = False,
        featurewise_std_normalization = False,
        rotation_range = rotation_range,
        width_shift_range = 0.02,
        height_shift_range = 0.02,
        vertical_flip = True,
        horizontal_flip = True,
        zoom_range = (0.8, 1.2),
        fill_mode='nearest'
        )
    return datagen


def image_transforme(datagen,image_content):
    image_content = np.expand_dims(image_content, axis=0)
    #expand dims
    datagen.fit(image_content)

    #Augmentation of the image
    X_augmented = datagen.flow(image_content, shuffle=False, batch_size=1)

    #Reduce dims
    X_aug = np.squeeze(X_augmented[0], axis=0)
    X_aug = X_aug.astype('uint8')


    return X_aug

def densenet201_preprocess(X,y):
    # expected transformation for 201 or 121 uses
    return preprocess_input(X),y
