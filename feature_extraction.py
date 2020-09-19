import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, models, applications, layers, optimizers, losses
import tensorflow as tf
import cv2

# feature extraction which will be used to construct feature maps

def prepare(file):
    IMG_SIZE = 160
    img_array = cv2.imread(file)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    #cv2.imshow("test",new_array)
    return new_array.reshape(1, IMG_SIZE, IMG_SIZE, 3)

# classes is number of classes
class VGG(tf.keras.Model):
    def __init__(self, classes):
        super(VGG, self).__init__()

        self.conv1_1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')
        self.conv1_2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')
        self.max1 = layers.MaxPool2D((2, 2), strides=(2, 2), name='block1_pool')

        # Block 2
        self.conv2_1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')
        self.conv2_2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')
        self.max2 = layers.MaxPool2D((2, 2), strides=(2, 2), name='block2_pool')

        # Block 3
        self.conv3_1 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')
        self.conv3_2 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')
        self.conv3_3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')
        self.max3 = layers.MaxPool2D((2, 2), strides=(2, 2), name='block3_pool')

        # Block 4
        self.conv4_1 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')
        self.conv4_2 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')
        self.conv4_3 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')
        self.max4 = layers.MaxPool2D((2, 2), strides=(2, 2), name='block4_pool')

        # Block 5
        self.conv5_1 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')
        self.conv5_2 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')
        self.conv5_3 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')
        self.max5 = layers.MaxPool2D((2, 2), strides=(2, 2), name='block5_pool')

        # Classification
        # self.flatten = layers.Flatten(name="flatten")
        # self.dense1 = layers.Dense(4096, activation='relu', name="dense1")
        # self.dense2 = layers.Dense(4096, activation='relu', name="dense2")
        # self.dense3 = layers.Dense(1000, activation='relu', name="dense3")
        # self.dense4 = layers.Dense(classes, activation='relu', name="softmax")

    def call(self,image):
        first = self.conv1_1(image)
        x = self.conv1_2(first)
        x = self.max1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.max2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.max3(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.max4(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        results = self.conv5_3(x)

        # Classification
        # x = self.flatten(results)
        # x = self.dense1(x)
        # x = self.dense2(x)
        # x = self.dense3(x)
        # results = self.dense4(x)
        return results


if __name__ == '__main__':

    arg = '2frame0.jpg' #your image path

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    image = prepare(arg)

    # base_model = applications.MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
    # base_model.trainable = False
    feature_maps = tf.placeholder(tf.float32, shape=image.shape)
    base_model = VGG(10)
    #design of actual NN
    model = models.Sequential([base_model])

    model.compile(optimizer='adam',
              loss=losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

    history = model.fit(train_images, train_labels,
                        validation_data=(test_images, test_labels))

    model.summary()
    # values = model.predict(image)

    iter = 0
    while(True):
        plt.figure()
        plt.imshow(values[0,:,:,iter].astype('uint8'))
        plt.colorbar()
        plt.grid(False)
        plt.show()
        iter = iter + 1
