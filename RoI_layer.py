from RoI import ROIPoolingLayer
from feature_extraction import prepare
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras import models, applications, layers, losses, optimizers, metrics


# input is set of regions array such as [0.5,0.2,0.7,0.4]
# other input is set of features

batch_size = 1
img_height = 5
img_width = 5
n_channels = 1
n_rois = 1
pooled_height = 2
pooled_width = 2

class MyDenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(MyDenseLayer, self).__init__()
    self.num_outputs = num_outputs

  def call(self, input):
    print("success")
    return input

if __name__ == '__main__':

    # roiss = tf.placeholder(tf.float32, shape=(1,4))
    # feature_maps = tf.placeholder(tf.float32, shape=(1,5))
    #
    # feature_input = tf.keras.Input(tensor = feature_maps)
    # roi_input = tf.keras.Input( tensor = roiss)
    #
    # dense = MyDenseLayer(5)(feature_maps)
    # test = tf.keras.Model(inputs=(feature_maps), outputs=dense)
    # input1 = np.array([[0],[0],[0],[0]])
    # input1 = input1.reshape(1, 4)
    #
    # input2 = np.array([[0],[0],[0],[0],[0]])
    # input2 = input2.reshape(1, 5)
    # print(input2.shape)
    #
    # with tf.Session() as session:
    #     result = session.run(dense, feed_dict={feature_input:input2})
    #

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
    # roiss = tf.placeholder(tf.float32, shape=(batch_size, n_rois, 4))
    # feature_maps = tf.placeholder(tf.float32, shape=values.shape)


    feature_input = tf.keras.Input(shape=values.shape, name = 'feature_input')
    roi_input = tf.keras.Input(shape=(batch_size, n_rois, 4), name = 'roi_input')

    roi = ROIPoolingLayer(pooled_height, pooled_width)([feature_input[0],roi_input[0]])
    fc1 = layers.Flatten()(roi)
    dense1 = layers.Dense(5*5*1280, activation='softmax', name = "class_output")(fc1)
    dense2 = layers.Dense(1, activation='sigmoid', name = "bounding_box")(fc1) # later change this into bounding box regression

    values = model.predict(image)
    values1 = maxpoolmodel.predict(image)

    region_array = np.asarray([[[0.0,0.0,1.0,1.0]]], dtype='float32')


    roimodel = tf.keras.Model(inputs=(feature_input, roi_input), outputs=(dense1, dense2))
    roimodel.compile(
        optimizer = optimizers.RMSprop(1e-3),
        loss={
            "bounding_box": losses.MeanSquaredError(),
            "class_output": losses.CategoricalCrossentropy(),
        },
        metrics={
        "bounding_box": [
            metrics.MeanAbsolutePercentageError(),
            metrics.MeanAbsoluteError(),
        ],
        "class_output": [metrics.CategoricalAccuracy()],
    },
    )
    roimodel.summary()
    values = values.reshape(1,  1, 5, 5, 1280) # take into account batch size which is first input
    region_array = region_array.reshape(1, 1, 1, 4)
    output2 = np.array([1])
    output1 = np.zeros(5*5*1280)
    output1 = output1.reshape(1, 5*5*1280)
    #roimodel.evaluate(x = [values,region_array], y = {"class_output":output1,"bounding_box":output2}  )
    print(roimodel.predict(x = [values,region_array]))
    iter = 0
    while(True):
        plt.figure()
        plt.imshow(values[0,0,:,:,iter].astype('uint8'))
        plt.colorbar()
        plt.grid(False)
        plt.figure()
        plt.imshow(values1[0,:,:,iter].astype('uint8'))
        plt.colorbar()
        plt.grid(False)
        plt.figure()
        plt.imshow(result[0,0,:,:,iter].astype('uint8')) # to be fixed later
        plt.colorbar()
        plt.grid(False)
        plt.show()
        iter = iter + 1
