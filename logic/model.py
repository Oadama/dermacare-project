import numpy as np
# from tensorflow import keras
from keras import Model, Sequential, layers
from keras.callbacks import EarlyStopping,CSVLogger, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
import os
from logic.preprocessing import data_augmentation
from params import *
from keras.applications.densenet import DenseNet201, DenseNet121

def initialize_model(num_classes:int,
                    model_type:str=MODEL_TYPE,
                    kernel_size:int=3,
                    val_dropout:float=0.2,
                    img_height:int=IMAGE_HEIGHT,
                    img_width:int=IMAGE_WIDTH,
                    input_shape:tuple=None,
                    ) -> Model:
    """
    Initialize adequate CNN model function of model_type
    """
    if str.upper(model_type) == 'DERMA':
        model = Sequential([
        data_augmentation(img_height,img_width, val_rotation=0.1, val_zoom=0.1),
        layers.Rescaling(1./255),
        layers.Conv2D(16, kernel_size, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, kernel_size, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, kernel_size, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(val_dropout),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
        ])
    elif str.upper(model_type) == 'DENSENET201':
        model = DenseNet201(
                            include_top=False,
                            weights="imagenet",
                            input_shape=input_shape,
                            classifier_activation="softmax"
                            )
        model.trainable = False
        model = Sequential([
        #model,
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
        ])
    elif str.upper(model_type) == 'DENSENET121':

        model_d=DenseNet121(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
                )

        model= model_d.output
        model= layers.GlobalAveragePooling2D()(model)
        model= layers.BatchNormalization()(model)
        model= layers.Dropout(0.5)(model)
        model= layers.Dense(1024,activation='relu')(model)
        model= layers.Dense(512,activation='relu')(model)
        model= layers.BatchNormalization()(model)
        model= layers.Dropout(0.5)(model)

        preds=layers.Dense(8,activation='softmax')(model)
        model=Model(inputs=model_d.input, outputs=preds)

        # freeze all layers except for the last 8
        for layer in model.layers[:-8]:
            layer.trainable=False

        for layer in model.layers[-8:]:
            layer.trainable=True

    else:
        print(f'\n❌ Unknown model type')

    print(f'✅ Model initialized')

    return model

def compile_model(model:Model, model_type:str=MODEL_TYPE) -> Model:
    """
    Compile the defined CNN model
    """

    if model_type.upper() == 'DERMA':
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        opt='adam'
        metrics=['accuracy']
    elif  model_type.upper() == 'DENSENET201':
        loss='categorical_crossentropy'
        opt =tf.keras.optimizers.Adam(learning_rate=0.001)
        metrics=['accuracy']
    elif model_type.upper() == 'DENSENET121':
        loss='categorical_crossentropy'
        opt='adam'
        metrics=['accuracy', tf.keras.metrics.Recall()]

    model.compile(optimizer=opt,
              loss=loss,
              metrics=metrics)

    print(f'✅ Model compiled')
    return model

def train_model(
        model:Model,
        X:np.ndarray,
        validation_data,
        model_type:str=MODEL_TYPE,
        batch_size=BATCH_SIZE,
        epochs:int=64,
        patience:int=2,
        verbose:int=0,
        factor:float=0.5,
        min_lr:float=1e-3,
        save_best_only:bool=True,
        restore_best_weights:bool=True
    ) :

    """
    Fit the model and return a tuple (fitted_model, history)
    """

    if str.upper(model_type) == 'DERMA':
        monitor="val_loss"
        mode='max'
    elif str.upper(model_type) in ['DENSENET201','DENSENET121']:
        monitor = 'val_accuracy'
        mode='max'

    es = EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=patience,
        restore_best_weights=restore_best_weights,
        verbose=verbose
    )
    csv_logger = CSVLogger(f'{model_type}_training.log')

    checkpoint_callback = ModelCheckpoint(
                filepath=LOCAL_CHECKPOINT_PATH,
                save_weights_only=True,
                monitor=monitor,
                mode=mode,
                save_best_only=True
                )

    if str.upper(model_type) == 'DENSENET121':
        anne = ReduceLROnPlateau(
                        monitor=monitor,
                        factor=factor,
                        patience=patience,
                        verbose=verbose,
                        min_lr=min_lr
                        )
        checkpoint = ModelCheckpoint(
                    f'{LOCAL_CHECKPOINT_PATH}/{model_type}_model.h5',
                    verbose=verbose,
                    save_best_only=save_best_only
                    )
        callbacks=[anne,checkpoint]
    else:
        callbacks=[es, csv_logger,checkpoint_callback]


    history = model.fit(X,
                        validation_data=validation_data,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks
    )
    # model_type,

    print(f'✅ Model succesfully trained through {len(history.epoch)} epochs')

    return model, history

def evaluate_model(
        model: Model,
        X: np.ndarray,
        batch_size=BATCH_SIZE
    ) :
    """
    Evaluate trained model performance on the dataset
    """

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        batch_size=batch_size,
        verbose=0,
        return_dict=True
    )

    print(f'✅ Model evaluated, [Loss, Accuracy, Recall]: [{round(metrics["loss"], 2)},{round(metrics["accuracy"], 2)},{round(metrics.get("accuracy"), 2)}]')

    return metrics
