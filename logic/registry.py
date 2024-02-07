import glob
import os
import time

from tensorflow import keras
from google.cloud import storage
from params import *
from logic.model import compile_model
from google.oauth2 import service_account



def save_model(model: keras.Model = None,model_type:str=MODEL_TYPE) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
    - if MODEL_TARGET='mlflow', also persist it on MLflow instead of GCS (for unit 0703 only) --> unit 03 only
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{model_type}_{timestamp}.{MODEL_SUFFIX}")
    model.save(model_path)

    print("✅ Model saved locally")

    if MODEL_TARGET == "gcs":

        model_filename = model_path.split("/")[-1]
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to GCS")

        return None
    return None


def load_model(stage="Production", model_type:str=MODEL_TYPE, compile:bool=True) -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model is found

    """
    ## print(MODEL_TARGET)

    if MODEL_TARGET == "local":
        print("\nLoad latest model from local registry...")

        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH)
        local_model_paths = glob.glob(f"model/models/{model_type}_*{MODEL_SUFFIX}")
        print(local_model_paths)
        if not local_model_paths:
            print(f"\n❌ repo not found {local_model_paths}")
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]
        print(most_recent_model_path_on_disk)
        print("\nLoad latest model from disk...")

        latest_model = keras.models.load_model(most_recent_model_path_on_disk, compile=False)
        latest_model = compile_model(latest_model, MODEL_TYPE)

        print("✅ Model loaded from local disk")


        return latest_model

    elif MODEL_TARGET == "gcs":
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!
        print("\nLoad latest model from GCS...")


        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))
        print(blobs)
        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
            latest_blob.download_to_filename(latest_model_path_to_save)
            print(latest_model_path_to_save)
            latest_model = keras.models.load_model(latest_model_path_to_save, compile=compile)
            #latest_model = compile_model(latest_model, MODEL_TYPE)

            print("✅ Latest model downloaded from cloud storage")

            return latest_model
        except:
            print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

            return None
    else:
        print(f"\n❌ No model found where asked")
        return None
