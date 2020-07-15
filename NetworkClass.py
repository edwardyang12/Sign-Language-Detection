from RoI import ROIPoolingLayer
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras import datasets, models, applications, layers, losses, optimizers, metrics
from feature_extraction import prepare

sess = tf.InteractiveSession()
batch_size = 1
img_height = 5
img_width = 5
n_channels = 1
n_rois = 1
pooled_height = 2
pooled_width = 2

class ROIModel(tf.keras.Model):
    def __init__(self, roiss, feature_maps):
        super(ROIModel, self).__init__()
        self.CNN = applications.MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
        self.feature_input = tf.keras.Input(tensor = feature_maps, name = 'feature_input')
        self.roi_input = tf.keras.Input(tensor = roiss, name = 'roi_input')
        self.roi = ROIPoolingLayer(pooled_height, pooled_width)([self.feature_input,self.roi_input])
        self.fc1 = layers.Flatten()
        self.dense1 = layers.Dense(32, activation='softmax', name = 'class_output')
        self.dense2 = layers.Dense(32, activation='sigmoid', name = 'bounding_box') # later change this into bounding box regression

    def call(self, image):

        features = self.CNN.predict()
        region_array = np.asarray([[[0.0,0.0,1.0,1.0]]], dtype='float32')
        result = sess.run(self.roi, feed_dict={self.feature_input:features, self.roi_input:region_array})
        flatten = self.fc1(result)
        return self.dense1(flatten), self.dense2(flatten)


if __name__ == '__main__':
    arg = '2frame0.jpg' #your image path

    image = prepare(arg)

    base_model = applications.MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')

    #base_model.summary()

    image = tf.convert_to_tensor(image)
    print(type(image))
    # image = image.eval()
    # print(type(image))
    # print(image)

    values = base_model.predict(x = image)
    # values = base_model.evaluate(x=data_image, y=values ) # just to get shape

    roiss = tf.placeholder(tf.float32, shape=(batch_size, n_rois, 4))
    feature_maps = tf.placeholder(tf.float32, shape=values.shape)
    model = ROIModel(roiss, feature_maps)

    model.compile(
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

    model.build(image.shape)
    model.summary()
    #model.evaluate(x = data_image)

    print("sucess")
