import cv2
import tensorflow as tf
import numpy as np
import sys

CATEGORIES = ["five_finger","one_finger","four_finger","middle_finger","two_finger","three_finger"]
arg = sys.argv[1]

def prepare(file):
    IMG_SIZE = 50
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
model = tf.keras.models.load_model("CNN.model")
#image = "2frame0.jpg" #your image path
image = prepare(arg)

prediction = model.predict([image])
prediction = list(prediction[0])
print()
print("----------------" + CATEGORIES[prediction.index(max(prediction))]+ "-----------------")

