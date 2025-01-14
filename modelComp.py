import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras 
from PIL import Image

#VGG16 model 
vggmodel = tf.keras.models.load_model("modelT.h5")

#CNN model
pbl = tf.keras.models.load_model("pbl.h5")

image_path = input("Enter image path!")
img = Image.open(image_path)

if img.mode != "RGB":
    img = img.convert("RGB")

img_resized = img.resize((150, 150))

img_array = np.array(img_resized)

img_array = img_array / 255.0

img_array = np.expand_dims(img_array, axis=0)

class_labels = ["Normal", "Pneumonia"]

predictionsvgg = vggmodel.predict(img_array)
print("VGG16 model's predictions !")
predicted_class = 0 if predictionsvgg[0][0] < 0.5 else 1
print(predictionsvgg)
output_label = class_labels[predicted_class]
print(f"Prediction: {output_label}")


predictionspbl = pbl.predict(img_array)
print("CNN model's predictions !")
predicted_class = 0 if predictionspbl[0][0] < 0.5 else 1
print(predictionspbl)
output_label = class_labels[predicted_class]
print(f"Prediction: {output_label}")