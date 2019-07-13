from tensorflow.python import keras
from tensorflow.python.keras import layers
from pyranet.layers.utils import *
from pyranet.models.pyranet import *
from pyranet.layers.variables import *


class Pooling3D(layers.Layer):

    def __init__(self, rf=(3, 2, 2), strides=(1, 1, 2, 2, 1), activation=keras.layers.LeakyReLU(alpha=0.1),
                 kernel_initializer=None, bias_initializer=None, weight_decay=None,
                 padding="VALID", data_format="NDHWC", log=False, name="pooling_3d_layer", **kwargs):
        super(Pooling3D, self).__init__(**kwargs)

        self.activation = activation

        self.rf = check_receptive_field(rf, 3)
        self.strides = check_strides(strides, 3)

        self.kernel_initializer = check_variable_initializer(kernel_initializer, default='XAVIER')
        self.bias_initializer = check_variable_initializer(bias_initializer, default='XAVIER')
        self.weight_decay = weight_decay

        self.padding = padding
        self.data_format = data_format
        self._name = name

        self.kernel = None
        self.bias = None

        self.log = log

    def build(self, input_shape):
        # Make sure to call the `build` method at the end
        super(Pooling3D, self).build(input_shape)

    def call(self, inputs, **kwargs):
        kernel_shape = pool3d_weight_initializer_size_like(inputs)
        bias_shape = pool3d_bias_initializer_size_like(inputs)

        # Move kernel and bias in build method
        if self.kernel is None:
            self.kernel = self.add_weight(name='kernel',
                                          shape=kernel_shape,
                                          initializer=self.kernel_initializer,
                                          regularizer=keras.regularizers.l2(self.weight_decay)
                                          if self.weight_decay else None,
                                          trainable=True)
        if self.bias is None:
            self.bias = self.add_weight(name='bias',
                                        shape=bias_shape,
                                        initializer=self.bias_initializer,
                                        regularizer=keras.regularizers.l2(self.weight_decay)
                                        if self.weight_decay else None,
                                        trainable=True)

        x = tf.multiply(inputs, self.kernel, name="mul_weights")
        x = tf.add(x, self.bias, name="bias_add")

        if self.activation:
            x = self.activation(x)

        if self.log:
            tf.logging.info("{} | Weight depth: {} - strides: {}".format(self._name, self.rf[0], self.strides[1:-1]))
            tf.logging.info("\t{} {}".format(self.kernel.name, self.kernel.shape))
            tf.logging.info("\t{} {}".format(self.bias.name, self.bias.shape))
            tf.logging.info("\t{} {}".format(self.x.name, self.x.shape))

        return x


class MaxPooling3D(Pooling3D):

    def build(self, input_shape):
        # Make sure to call the `build` method at the end
        super(MaxPooling3D, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if isinstance(inputs.shape, tf.TensorShape):
            _, d, h, w, c = [x.value for x in inputs.shape]
        else:
            _, d, h, w, c = inputs.shape

        weight_depth = self.rf[0]

        out_depth, _, _, _ = [int(x) for x in ws3d_layer_output_shape(input_shape=(d, h, w, c),
                                                                      rf=self.rf, strides=self.strides)]
        pool = tf.nn.max_pool3d(inputs, self.strides, self.strides, padding=self.padding,
                                data_format=self.data_format, name="max_pooling3d")

        output = []
        for depth in range(out_depth):
            # Please double check here
            output.append(tf.reduce_max(pool[:, depth:depth + weight_depth], axis=1))

        output = tf.transpose(output, [1, 0, 2, 3, 4])
        return super(MaxPooling3D, self).call(output, **kwargs)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, tf.TensorShape):
            input_shape = tuple(x.value for x in input_shape)
        shape = (None,) + pool3d_layer_output_shape(input_shape[1:], rf=self.rf,
                                                    strides=self.strides, padding=self.padding)
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(MaxPooling3D, self).get_config()

        base_config['activation'] = self.activation

        base_config['rf'] = self.rf
        base_config['strides'] = self.strides

        base_config['kernel_initializer'] = self.kernel_initializer
        base_config['bias_initializer'] = self.bias_initializer
        base_config['weight_decay'] = self.weight_decay

        base_config['padding'] = self.padding
        base_config['data_format'] = self.data_format
        base_config['name'] = self._name

        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
