from tensorflow.keras import datasets, layers, losses, optimizers, metrics, Input, Model
import tensorflow as tf
from feature_extraction import prepare, VGG
import roi_helpers
import config
from tensorflow.keras import backend as K
import cv2

class RPN(tf.keras.Model):
    def __init__(self, num_anchors):
        super(RPN,self).__init__()
        self.x = layers.Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')
        self.x_class = layers.Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')
        self.x_regr = layers.Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')

    def call(self, features):
        x = self.x(features)
        classes = self.x_class(x)
        regression = self.x_regr(x)
        return [classes, regression, features]

# regions should be a set of regions
def view_region(image,regions):
    row,col = image.shape[0:2]
    for region in regions:
        cv2.rectangle(image, (row*region[0], col*region[1]), (row*region[2], col*region[3]), (0, 0, 255))
    cv2.imshow('img', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    C = config.Config()
    arg = '2frame0.jpg'
    test_image =  prepare(arg)
    # (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    # train_images, test_images = train_images / 255.0, test_images / 255.0
    input_shape_img = (None,None,3)

    img_input = Input(shape=input_shape_img)
    anchor_box_scales = [128, 256, 512]
    anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]
    num_anchors = len(anchor_box_scales) * len(anchor_box_ratios)

    features = VGG(10)(img_input)
    rpn = RPN(num_anchors)(features)

    model = Model(inputs = img_input, outputs= rpn)
    model.compile(optimizer='sgd', loss='mse')

    test = test_image.reshape(1,160,160,3)

    values = model.predict(test)

    R = roi_helpers.rpn_to_roi(values[0], values[1], C, K.image_data_format(), use_regr=True, overlap_thresh=0.7, max_boxes=300)
    print(R)
    #view_region(test_image,R)
