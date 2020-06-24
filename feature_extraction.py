import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, models, layers, losses, applications, utils
import cv2

# feature extraction which will be used to construct feature maps

def prepare(file):
    IMG_SIZE = 160
    img_array = cv2.imread(file)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    cv2.imshow("test",new_array)
    return new_array.reshape(1, IMG_SIZE, IMG_SIZE, 3)


if __name__ == '__main__':

    arg = '2frame0.jpg' #your image path

    image = prepare(arg)

    base_model = applications.MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    #design of actual NN
    model = models.Sequential([base_model])

    values = model.predict(image)

    iter = 0
        while(True):
        plt.figure()
        plt.imshow(values[0,:,:,iter].astype('uint8'))
        plt.colorbar()
        plt.grid(False)
        plt.show()
        iter = iter + 1
