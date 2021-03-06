import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.layers import Layer
from tensorflow.keras.backend import placeholder
from tensorflow.keras import datasets
import numpy as np
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()
class ROIPoolingLayer(Layer):
    """ Implements Region Of Interest Max Pooling
        for channel-first images and relative bounding box coordinates

        # Constructor parameters
            pooled_height, pooled_width (int) --
              specify height and width of layer outputs

        Shape of inputs
            [(batch_size, pooled_height, pooled_width, n_channels),
             (batch_size, num_rois, 4)]

        Shape of output
            (batch_size, num_rois, pooled_height, pooled_width, n_channels)

    """
    def __init__(self, pooled_height, pooled_width, **kwargs):
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width

        super(ROIPoolingLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        """ Returns the shape of the ROI Layer output
        """
        feature_map_shape, rois_shape = input_shape
        assert feature_map_shape[0] == rois_shape[0]
        batch_size = feature_map_shape[0]
        n_rois = rois_shape[1]
        n_channels = feature_map_shape[3]
        return (batch_size, n_rois, self.pooled_height,
                self.pooled_width, n_channels)

    def call(self, x):
        x[0] = tf.dtypes.cast(x[0], tf.float32)
        """ Maps the input tensor of the ROI layer to its output

            # Parameters
                x[0] -- Convolutional feature map tensor,
                        shape (batch_size, pooled_height, pooled_width, n_channels)
                x[1] -- Tensor of region of interests from candidate bounding boxes,
                        shape (batch_size, num_rois, 4)
                        Each region of interest is defined by four relative
                        coordinates (x_min, y_min, x_max, y_max) between 0 and 1
            # Output
                pooled_areas -- Tensor with the pooled region of interest, shape
                    (batch_size, num_rois, pooled_height, pooled_width, n_channels)
        """
        def curried_pool_rois(x):
          return ROIPoolingLayer._pool_rois(x[0], x[1],
                                            self.pooled_height,
                                            self.pooled_width)

        pooled_areas = tf.map_fn(curried_pool_rois, x, dtype=tf.float32)

        return pooled_areas

    @staticmethod
    def _pool_rois(feature_map, rois, pooled_height, pooled_width):
        """ Applies ROI pooling for a single image and varios ROIs
        """
        def curried_pool_roi(roi):
          return ROIPoolingLayer._pool_roi(feature_map, roi,
                                           pooled_height, pooled_width)

        pooled_areas = tf.map_fn(curried_pool_roi, rois, dtype=tf.float32)
        return pooled_areas

    @staticmethod
    def _pool_roi(feature_map, roi, pooled_height, pooled_width):
        """ Applies ROI pooling to a single image and a single region of interest
        """

        # Compute the region of interest
        feature_map_height = int(feature_map.shape[0])
        feature_map_width  = int(feature_map.shape[1])

        h_start = tf.cast(feature_map_height * roi[0], 'int32')
        w_start = tf.cast(feature_map_width  * roi[1], 'int32')
        h_end   = tf.cast(feature_map_height * roi[2], 'int32')
        w_end   = tf.cast(feature_map_width  * roi[3], 'int32')


        region = feature_map[h_start:h_end, w_start:w_end, :]


        # Divide the region into non overlapping areas
        region_height = h_end - h_start
        region_width  = w_end - w_start
        h_step = tf.cast( region_height / pooled_height, 'int32')
        w_step = tf.cast( region_width  / pooled_width , 'int32')

        areas = [[(
                    i*h_step,
                    j*w_step,
                    (i+1)*h_step if i+1 < pooled_height else region_height,
                    (j+1)*w_step if j+1 < pooled_width else region_width
                   )
                   for j in range(pooled_width)]
                  for i in range(pooled_height)]

        # take the maximum of each area and stack the result
        def pool_area(x):
          return tf.math.reduce_max(region[x[0]:x[2], x[1]:x[3], :], axis=[0,1])

        pooled_features = tf.stack([[pool_area(x) for x in row] for row in areas])
        return pooled_features

if __name__ == '__main__':
    # Define parameters
    batch_size = 1
    img_height = 200
    img_width = 100
    n_channels = 1
    n_rois = 3
    pooled_height = 6
    pooled_width = 6

    # Create feature map input
    # feature_maps_shape = (batch_size, img_height, img_width, n_channels)
    # feature_maps_tf = tf.placeholder(tf.float32, shape=feature_maps_shape)
    # feature_maps_np = np.ones(feature_maps_tf.shape, dtype='float32')
    # feature_maps_np[0, img_height-1, img_width-3, 0] = 50
    # print(f"feature_maps_np.shape = {feature_maps_np.shape}")

    # load data and preprocess it
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    one_image = train_images[0].reshape(1, 32, 32, 3)
    feature_maps_tf = tf.placeholder(tf.float32, shape=one_image.shape)

    # Create batch size
    roiss_tf = tf.placeholder(tf.float32, shape=(batch_size, n_rois, 4))
    roiss_np = np.asarray([[[0.5,0.2,0.7,0.4], [0.0,0.0,1.0,1.0],[0.6,0.0,0.0,1.0]]], dtype='float32')
    print(f"roiss_np.shape = {roiss_np.shape}")

    # Create layer
    roi_layer = ROIPoolingLayer(pooled_height, pooled_width)
    result = roi_layer([one_image,roiss_np])
    # pooled_features = roi_layer([feature_maps_tf, roiss_tf])
    # print(f"output shape of layer call = {pooled_features.shape}")
    # # Run tensorflow session
    # with tf.Session() as session:
    #     result = session.run(pooled_features,
    #                          feed_dict={feature_maps_tf:one_image,
    #                                     roiss_tf:roiss_np})
    #
    print(f"result.shape = {result.shape}")

    #print(one_image[0])
    plt.figure()
    plt.imshow(one_image[0])
    plt.colorbar()
    plt.grid(False)

    print(f"first  roi embedding=\n{result[0,0]}")
    plt.figure()
    plt.imshow(result[0,0].eval().astype('uint8'))
    plt.colorbar()
    plt.grid(False)

    print(f"second roi embedding=\n{result[0,1]}")
    plt.figure()
    plt.imshow(result[0,1].eval().astype('uint8'))
    plt.colorbar()
    plt.grid(False)
    plt.show()
