from RoI import ROIPoolingLayer
from feature_extraction import prepare
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras import models, applications, layers


# input is set of regions array such as [0.5,0.2,0.7,0.4]
# other input is set of features

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

    base_model1 = applications.MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
    base_model1.trainable = False


    #design of actual NN
    model = models.Sequential([base_model])

    maxpoolmodel = models.Sequential([base_model1])
    maxpoolmodel.add(layers.MaxPooling2D((2, 2)))

    values = model.predict(image) # just to get shape
    roiss = tf.placeholder(tf.float32, shape=(batch_size, n_rois, 4))
    feature_maps = tf.placeholder(tf.float32, shape=values.shape)

    roi = ROIPoolingLayer(pooled_height, pooled_width)([feature_maps,roiss])

    values = model.predict(image)
    values1 = maxpoolmodel.predict(image)

    region_array = np.asarray([[[0.0,0.0,1.0,1.0]]], dtype='float32')

    feature_input = tf.keras.Input(shape=(feature_maps.shape))
    roi_input = tf.keras.Input(shape=(roiss.shape))
    roimodel = tf.keras.Model(inputs=({feature_maps:feature_input, roiss:roi_input}), outputs=roi)
    #values2 = roi.predict(image)

    iter = 0
    while(True):
        plt.figure()
        plt.imshow(values[0,:,:,iter].astype('uint8'))
        plt.colorbar()
        plt.grid(False)
        plt.figure()
        plt.imshow(values1[0,:,:,iter].astype('uint8'))
        plt.colorbar()
        plt.grid(False)
        plt.show()
        iter = iter + 1
