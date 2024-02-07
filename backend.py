from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from logic.registry import load_model
import tensorflow as tf
from params import *
import numpy as np
import time
from keras.applications.densenet import preprocess_input
from logic.model import compile_model
import os

app = FastAPI()


model = load_model(compile=False)
model = compile_model(model, MODEL_TYPE)
app.state.model=model


# Allowing all middleware is optional, but good practice for dev purposes


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Predict end point
@app.post("/predict_cnn")

async def predict_cnn(img: UploadFile=File(...)):
    ### Receiving and decoding the image
    if app.state.model== None:
        return {'message': f"\n❌ No model found where asked"}

    contents = await img.read()
    timestamp = str(time.time()).replace(".","")
    file_name=f"{timestamp}.jpg"
    # os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "project_outputs")
    with open(file_name, "wb") as f:
        f.write(contents)
    img = tf.keras.utils.load_img(file_name, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    img_array = tf.keras.utils.img_to_array(img)
    img_array= preprocess_input(img_array)
    img_array = tf.expand_dims(img_array, 0)
    result=app.state.model.predict(img_array)
    categ=CLASS_NAMES[np.argmax(result)]
    prob=round(100*np.max(result),2)
    # msg=f'✅ This image most likely belongs to {categ} with a probability of {prob}%'
    msg=f'{categ} {prob}'
    # return {'message': msg, 'categ':categ, 'proba': result}
    return {'message': msg}
@app.get("/")
def root():
    return {'Dermasaaj': ' ✅ Here we are '}
