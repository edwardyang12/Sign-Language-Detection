import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
from RoI import ROIPoolingLayer
import numpy as np
from tensorflow.keras import datasets, models, applications, layers, losses, optimizers, metrics, Input, Model
from tensorflow.keras import backend as K
from feature_extraction import prepare, VGG
from RPN import RPN
import roi_helpers
import config
import losses as custom_losses
from datetime import datetime
import tensorboard
import data_generators
import pascal_voc_parser

pooled_height = 3
pooled_width = 3
anchor_box_scales = [128, 256, 512]
anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]
num_anchors = len(anchor_box_scales) * len(anchor_box_ratios)

class ROIModel(tf.keras.Model):
    def __init__(self,classes):
        super(ROIModel, self).__init__()

        self.VGG = VGG(classes)
        self.RPN = RPN(num_anchors)
        self.roi = ROIPoolingLayer(pooled_height, pooled_width)
        self.fc1 = layers.Flatten()
        self.dense1 = layers.Dense(classes, activation='softmax', name = "class_output") # number should be number of classes
        self.dense2 = layers.Dense((classes-1)*4, activation='linear', name = "bounding_box") # later change this into bounding box regression

    def call(self, image):
        # image = tf.dtypes.cast(image, tf.float32)
        features = self.VGG(image)
        classes, regression = self.RPN(features)

        regression1 = K.permute_dimensions(regression,(0, 3, 1, 2))
        regression1 = tf.reshape(regression1, [4,-1])
        regression1 = K.permute_dimensions(regression1,(1,0))

        all_probs = K.permute_dimensions(classes, (0, 3, 1, 2))
        all_probs = tf.reshape(all_probs, [-1])

        # batched
        # regression = K.permute_dimensions(regression,(0, 3, 1, 2))
        # regression = tf.reshape(regression, [36,4,-1])
        # regression = K.permute_dimensions(regression,(2,0,1))
        #
        # classes = K.permute_dimensions(classes, (0, 3, 1, 2))
        # all_probs = tf.reshape(classes, [36,-1])
        # all_probs = K.permute_dimensions(all_probs,(1,0))

        selected = tf.image.non_max_suppression(regression1, all_probs,300,0.9)
        regions = tf.gather(regression1,selected)
        regions = tf.expand_dims(regions,axis=0)

        result = self.roi([features,regions])

        flatten = self.fc1(result)
        flatten = tf.expand_dims(flatten,axis=0)
        flatten = K.permute_dimensions(flatten,(1,2,0))

        output = [self.dense1(flatten), self.dense2(flatten), regression, classes]

        return output

def get_img_output_length(width, height):
    def get_output_length(input_length):
        return input_length/16

    return get_output_length(width), get_output_length(height)

if __name__ == '__main__':

    # C = config.Config()
    # all_imgs = pascal_voc_parser.get_data_all(r"C:\Users\Edward\Desktop\VOC2012")
    # data_gen_train = data_generators.get_anchor_gt(all_imgs, 2, C, get_img_output_length, K.image_data_format(), mode='train')
    # print(data_gen_train[2])

    # arg = '2frame0.jpg' #your image path
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    #image = prepare(arg)

    temp_im_test = np.ones((50000,10,10))
    temp_im_train = np.ones((10000,10,10))
    temp_bb_train = np.ones((50000,10,36))
    temp_bb_test = np.ones((10000,10,36))
    temp_regr_train = np.ones((50000,2,2,36))
    temp_regr_test = np.ones((10000,2,2,36))
    temp_class_train = np.ones((50000,2,2,9))
    temp_class_test = np.ones((10000,2,2,9))

    model = ROIModel(10)

    model.compile(
        optimizer = optimizers.RMSprop(1e-3),
        loss={
            "output_1": custom_losses.class_loss_cls(),
            "output_2": custom_losses.class_loss_regr(9),
            "output_3": custom_losses.rpn_loss_regr(num_anchors),
            "output_4": custom_losses.rpn_loss_cls(num_anchors),
        },
        metrics={
            "output_1":'accuracy',
            "output_2":'accuracy',
            "output_3":'accuracy',
            "output_4":'accuracy',
    },
    )

    logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    # test_image = np.array(train_images[0]).reshape(1,32,32,3)
    # model.predict(test_image,callbacks=[tensorboard_callback])

    # for bounding and classifier
    history = model.fit(x=train_images, y=(temp_im_test,temp_bb_train, temp_regr_train,temp_class_train), epochs=2, batch_size = 1, verbose =1,
                        validation_data=(test_images, [temp_im_train,temp_bb_test,temp_regr_test,temp_class_test]))


    model.summary()
    # plt.plot(history.history['output_1_acc'], label='Classification')
    # plt.plot(history.history['output_2_mean_squared_error'], label ='Bounding Box Error')
    # plt.plot(history.history['val_output_1_acc'], label = 'val_accuracy')
    # plt.plot(history.history['val_output_2_mean_squared_error'], label = 'val_bounding_box')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.ylim([0.5, 1])
    # plt.legend(loc='lower right')
    # plt.show()
    #
