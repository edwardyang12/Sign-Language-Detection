from RoI import ROIPoolingLayer
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
tf.disable_v2_behavior()
from tensorflow.keras import datasets, models, applications, layers, losses, optimizers, metrics
from feature_extraction import prepare, VGG
#from RPN import RPN

pooled_height = 1
pooled_width = 1

class ROIModel(tf.keras.Model):
    def __init__(self,labels):
        super(ROIModel, self).__init__()
        # self.CNN = applications.MobileNetV2(input_shape=input_shape[1:], include_top=False, weights='imagenet')
        # self.CNN.trainable = False
        self.VGG = VGG(labels)
        self.roi = ROIPoolingLayer(pooled_height, pooled_width)
        self.fc1 = layers.Flatten()
        self.dense1 = layers.Dense(10, activation='softmax', name = "class_output") # number should be number of classes
        self.dense2 = layers.Dense(1, activation='sigmoid', name = "bounding_box") # later change this into bounding box regression

    def call(self, image):
        # image = tf.dtypes.cast(image, tf.float32)
        features = self.VGG(image)
        region_array = np.asarray([[[0.0,0.0,1.0,1.0]]], dtype='float32')
        result = self.roi([features,region_array])
        flatten = self.fc1(result)
        output = [self.dense1(flatten), self.dense2(flatten)]
        return output

if __name__ == '__main__':
    # arg = '2frame0.jpg' #your image path
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    #image = prepare(arg)

    temp_bb_train = np.ones(train_labels.shape[0])
    temp_bb_test = np.ones(test_labels.shape[0])

    model = ROIModel(10)

    model.compile(
        optimizer = optimizers.RMSprop(1e-3),
        loss={
            "output_2": losses.MeanSquaredError(),
            "output_1": losses.SparseCategoricalCrossentropy(),
        },
        metrics={
        "output_1": [
            'accuracy'
            #metrics.CategoricalAccuracy(),
        ],
        "output_2": [metrics.MeanSquaredError()],
    },
    )

    # for bounding and classifier
    history = model.fit(x=train_images, y=(train_labels, temp_bb_train), epochs=2, batch_size = 1, verbose =1,
                        validation_data=(test_images, [test_labels,temp_bb_test]))

    model.summary()
    plt.plot(history.history['output_1_acc'], label='Classification')
    plt.plot(history.history['output_2_mean_squared_error'], label ='Bounding Box Error')
    plt.plot(history.history['val_output_1_acc'], label = 'val_accuracy')
    plt.plot(history.history['val_output_2_mean_squared_error'], label = 'val_bounding_box')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    test = train_images[1]
    test = test.reshape(1, 32, 32, 3)
    print(model.predict(x=test))
    print(train_labels[1])

    # model.build([train_images[0].shape])
    # model.summary()
    # test = train_images[1]
    # test = test.reshape(1, 32, 32, 3)
    # print(test.shape)
    # print(model.predict(x=test))

    print("sucess")
