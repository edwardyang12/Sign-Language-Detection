from tensorflow.keras import datasets, models, applications, layers, losses, optimizers, metrics

class RPN(tf.keras.Model):
    def __init__(self,base_layers,num_anchors):
        super(RPN,self).__init__()
        self.x = Convolution2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')
        self.x_class = Convolution2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')
        self.x_regr = Convolution2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')

    def call(self, features):

        return [x_class, x_regr, base_layers]
