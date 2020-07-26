import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
tf.disable_v2_behavior()
from tensorflow.keras import datasets, models, applications, layers, losses, optimizers, metrics

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
#image = prepare(arg)


base_model = applications.MobileNetV2(input_shape=train_images.shape[1:], include_top=False, weights='imagenet')
base_model.trainable = False
model = models.Sequential([base_model])

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='adam',
              loss=losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

# graph the learning process
history = model.fit(train_images, train_labels, epochs=30,
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

plt.show()
