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
    def __init__(self):
        super(ROIModel, self).__init__()
        self.CNN = applications.MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
        self.roi = ROIPoolingLayer(pooled_height, pooled_width)
        self.fc1 = layers.Flatten()
        self.dense1 = layers.Dense(5*5*1280, activation='softmax', name = "class_output")
        self.dense2 = layers.Dense(1, activation='sigmoid', name = "bounding_box") # later change this into bounding box regression

    def call(self, image):
        image = tf.dtypes.cast(image, tf.float32)
        features = self.CNN(image)
        region_array = np.asarray([[[0.0,0.0,1.0,1.0]]], dtype='float32')
        result = self.roi([features,region_array])
        flatten = self.fc1(result)
        return [self.dense1(flatten), self.dense2(flatten)]

if __name__ == '__main__':
    arg = '2frame0.jpg' #your image path

    image = prepare(arg)

    base_model = applications.MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
    #base_model.summary()


    values = base_model.predict(image)

    model = ROIModel()

    model.compile(
        optimizer = optimizers.RMSprop(1e-3),
        loss={
            "output_1": losses.MeanSquaredError(),
            "output_2": losses.CategoricalCrossentropy(),
        },
        metrics={
        "output_1": [
            metrics.MeanAbsolutePercentageError(),
            metrics.MeanAbsoluteError(),
        ],
        "output_2": [metrics.CategoricalAccuracy()],
    },
    )

    model.build(image.shape)
    model.summary()
    print(model.predict(x=image))

    print("sucess")
