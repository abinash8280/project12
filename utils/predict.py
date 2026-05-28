import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("model/soil_model.h5")

soil_classes = ["alluvial", "black", "clay", "red"]

def predict_soil(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0

    prediction = model.predict(img)
    index = np.argmax(prediction)

    return soil_classes[index]




