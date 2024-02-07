from logic.preprocessing import initialize_dataset_from_file, get_split_image_data,densenet201_preprocess, class_names
from logic.model import initialize_model, compile_model, train_model, evaluate_model
from logic.registry import save_model
from utils import plot_loss_accuracy
from pathlib import Path
from params import *

# Extract if need be dataset from archive
# original or cropped and augmented dataset


if str.upper(ARCHIVE_EXTRACT) == 'YES':
    parent_path = Path("../root/.keras/datasets")

    if 0 != len(ARCHIVE_PARENT_FOLDER):
        #parent_path+=f'/{ARCHIVE_PARENT_FOLDER}'
        parent_path = Path(f"../root/.keras/datasets/{ARCHIVE_PARENT_FOLDER}")

    if not parent_path.is_dir():
        file_name=f'{PATH_URL_ARCHIVE_FILE}/{ARCHIVE_FILE}'
        parent_path=initialize_dataset_from_file(file_name,extract=True,archive_format='zip')
else:
    # implement here the image processing
    if str.upper(IMAGES_CROP) == 'YES':
        # report here jeremy's codes
        print('Missing cropping codes')
    if str.upper(IMAGES_AUGMENT) == 'YES':
        # report here adama's codes
        print('Missing augmentation codes')

    parent_path=OUTPUT_PARENT_FOLDER


# initialize tensorflow dataset
# & calibrate data structure


img_height=IMAGE_HEIGHT
img_width=IMAGE_WIDTH
batch_size=BATCH_SIZE


# Split and prepare inputs for the model
child_path='train'
train_ds=get_split_image_data(parent_path, child_path, img_height, img_width,batch_size )
child_path='test'
test_ds=get_split_image_data(parent_path, child_path, img_height, img_width,batch_size )
child_path='valid'
val_ds=get_split_image_data(parent_path, child_path, img_height, img_width,batch_size )


#Class_names
num_classes = len(class_names(train_ds))



# initialize and finetune the CNN model
kernel_size=3
val_dropout=0.2
model=initialize_model(num_classes, MODEL_TYPE,  kernel_size, val_dropout,img_height, img_width)
# compile the model
model=compile_model(model, MODEL_TYPE)
#train the model
patience=2
verbose=1
epochs=1


if str.upper(MODEL_TYPE) in ['DENSENET201', 'DENSENET121']:
    train_ds = train_ds.map(densenet201_preprocess)
    val_ds = val_ds.map(densenet201_preprocess)
    test_ds = test_ds.map(densenet201_preprocess)


model, history=train_model(
        model_type=MODEL_TYPE,
        model=model,
        X=train_ds,
        validation_data=val_ds,
        batch_size=batch_size,
        patience=patience,
        verbose=verbose,
        factor=0.5,
        min_lr=1e-3,
        save_best_only=True,
        restore_best_weights=True
        )
# save the model
save_model(model_type=MODEL_TYPE, model=model)  # uses LOCAL_REGISTRY_PATH


#plot_loss_accuracy(history, epochs=epochs)


#evaluate the model with the test data set
metrics=evaluate_model(
        model,
        test_ds,
        batch_size=BATCH_SIZE
        )
