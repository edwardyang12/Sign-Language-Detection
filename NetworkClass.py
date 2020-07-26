from RoI import ROIPoolingLayer
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras import datasets, models, applications, layers, losses, optimizers, metrics
from feature_extraction import prepare

pooled_height = 1
pooled_width = 1

class ROIModel(tf.keras.Model):
    def __init__(self, input_shape):
        super(ROIModel, self).__init__()
        self.CNN = applications.MobileNetV2(input_shape=input_shape[1:], include_top=False, weights='imagenet')
        self.CNN.trainable = False
        self.roi = ROIPoolingLayer(pooled_height, pooled_width)
        self.fc1 = layers.Flatten()
        self.dense1 = layers.Dense(10, activation='softmax', name = "class_output") # number should be number of classes
        self.dense2 = layers.Dense(1, activation='sigmoid', name = "bounding_box") # later change this into bounding box regression

    def call(self, image):
        image = tf.dtypes.cast(image, tf.float32)
        features = self.CNN(image)
        region_array = np.asarray([[[0.0,0.0,1.0,1.0]]], dtype='float32')
        result = self.roi([features,region_array])
        flatten = self.fc1(result)
        return [self.dense1(flatten), self.dense2(flatten)]

if __name__ == '__main__':
    # arg = '2frame0.jpg' #your image path
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    #image = prepare(arg)

    temp_bb_train = np.ones(train_labels.shape[0])
    temp_bb_test = np.ones(test_labels.shape[0])

    model = ROIModel(train_images.shape)

    model.compile(
        optimizer = optimizers.RMSprop(1e-3),
        loss={
            "output_1": losses.MeanSquaredError(),
            #"output_2": losses.CategoricalCrossentropy(),
        },
        metrics={
        "output_1": [
            metrics.MeanAbsolutePercentageError(),
        ],
        #"output_2": [metrics.CategoricalAccuracy()],
    },
    )

    print(train_images.shape)
    # for bounding and classifier
    history = model.fit(train_images, [train_labels, temp_bb_train], epochs=30,
                        validation_data=(test_images, [test_labels,temp_bb_test]))

    # model.build([train_images[0].shape])
    model.summary()
    # test = train_images[0]/255.0
    # test = test.reshape(1, 32, 32, 3)
    # print(test.shape)
    # print(model.predict(x=test))

    print("sucess")
