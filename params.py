import os

##################  VARIABLES  ##################

IMAGE_HEIGHT=int(os.environ.get('IMAGE_HEIGHT'))
#IMAGE_HEIGHT=os.environ.get('IMAGE_HEIGHT')
IMAGE_WIDTH=int(os.environ.get('IMAGE_WIDTH'))
BATCH_SIZE=int(os.environ.get('BATCH_SIZE'))
BUCKET_NAME = os.environ.get("BUCKET_NAME")
MODEL_TARGET = os.environ.get("MODEL_TARGET")
PATH_URL_ARCHIVE_FILE=os.environ.get("PATH_URL_ARCHIVE_FILE")
ARCHIVE_EXTRACT=os.environ.get("ARCHIVE_EXTRACT")
ARCHIVE_FILE=os.environ.get("ARCHIVE_FILE")
ARCHIVE_PARENT_FOLDER=os.environ.get("ARCHIVE_PARENT_FOLDER")

IMAGES_CROP=os.environ.get("IMAGES_CROP")
IMAGES_AUGMENT=os.environ.get("IMAGES_AUGMENT")
OUTPUT_PARENT_FOLDER=os.environ.get("OUTPUT_PARENT_FOLDER")
MODEL_TYPE=os.environ.get("MODEL_TYPE").strip(' ')
MODEL_SUFFIX=os.environ.get("MODEL_SUFFIX").strip(' ')


##################  CONSTANTS  #####################
CLASS_NAMES=['AK', 'BCC','BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']

# LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "project_outputs")
#LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".code", "oadama", "dermasaaj_project", "model")

LOCAL_REGISTRY_PATH = "model"
LOCAL_CHECKPOINT_PATH =  os.path.join(os.path.expanduser('~'), ".code","oadama", "dermasaaj_project", "project_outputs","checkpoint")


GCP_PROJECT_ID=os.environ.get("GCP_PROJECT_ID").strip(' ')
DOCKER_IMAGE_NAME=os.environ.get("DOCKER_IMAGE_NAME").strip(' ')
GCR_MULTI_REGION=os.environ.get("GCR_MULTI_REGION").strip(' ')
GCR_REGION=os.environ.get("GCR_REGION").strip(' ')
