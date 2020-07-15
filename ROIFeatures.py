from RoI import ROIPoolingLayer
from feature_extraction import prepare
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras import models, applications

batch_size = 1
img_height = 5
img_width = 5
n_channels = 1
n_rois = 1
pooled_height = 5
pooled_width = 5

if __name__ == '__main__':

    arg = '2frame0.jpg' #your image path

    image = prepare(arg)

    base_model = applications.MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    #design of actual NN
    model = models.Sequential([base_model])

    values = model.predict(image)

    # Create batch size
    roiss_np = np.asarray([[[0.0,0.0,1.0,1.0]]], dtype='float32')
    print(f"roiss_np.shape = {roiss_np.shape}")

    roi_layer = ROIPoolingLayer(pooled_height, pooled_width)
    result = roi_layer([values,roiss_np])

    print(f"result.shape = {result.shape}")
    iter = 0
    while(True):
        #print(test_image[0])
        plt.figure()
        plt.imshow(values[0,:,:,iter])
        plt.colorbar()
        plt.grid(False)

        print(f"first  roi embedding=\n{result[0,0]}")
        plt.figure()
        plt.imshow(result[0,0,:,:,iter].eval().astype('uint8'))
        plt.colorbar()
        plt.grid(False)
        plt.show()
        iter = iter + 1
