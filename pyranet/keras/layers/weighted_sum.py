import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
from pyranet.layers.utils import *
from pyranet.models.pyranet import *


class WeightedSum3D(layers.Layer):

    def __init__(self, out_filters, rf=(3, 4, 4), strides=(1, 1, 1), activation=keras.layers.LeakyReLU(alpha=0.1),
                 kernel_initializer=None, bias_initializer=None, weight_decay=None, padding="VALID",
                 data_format="NDHWC", log=False, name="weighted_sum_3d_layer", **kwargs):
        super(WeightedSum3D, self).__init__(**kwargs)

        self.out_filters = out_filters
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
        if isinstance(input_shape, tf.TensorShape):
            _, d, h, w, c = [x.value for x in input_shape]
        else:
            _, d, h, w, c = input_shape
        kernel_shape = self.rf[0], h, w, c, self.out_filters
        bias_shape = self.compute_output_shape(input_shape).as_list()[1:]

        self.kernel = self.add_weight(name='kernel',
                                      shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      regularizer=keras.regularizers.l2(self.weight_decay)
                                      if self.weight_decay else None,
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=bias_shape,
                                    initializer=self.kernel_initializer,
                                    regularizer=keras.regularizers.l2(self.weight_decay)
                                    if self.weight_decay else None,
                                    trainable=True)

        # Make sure to call the `build` method at the end
        super(WeightedSum3D, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = ws3d_base(inputs, self.kernel, rf=self.rf, strides=self.strides,
                      padding=self.padding, data_format=self.data_format, name=self._name)

        x = tf.add(x, self.bias, name="bias_add")

        if self.activation:
            x = self.activation(x)

        if self.log:
            tf.logging.info("{} | RF: {} - strides: {}".format(self._name, self.rf, self.strides[1:-1]))
            tf.logging.info("\t{} {}".format(self.kernel.name, self.kernel.shape))
            tf.logging.info("\t{} {}".format(self.bias.name, self.bias.shape))
            tf.logging.info("\t{} {}".format(x.name, x.shape))

        return x

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, tf.TensorShape):
            input_shape = tuple(x.value for x in input_shape)
        shape = (None,) + ws3d_layer_output_shape(input_shape[1:-1] + (self.out_filters,), rf=self.rf,
                                                  strides=self.strides, padding=self.padding)
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(WeightedSum3D, self).get_config()

        # base_config['output_dim'] = self.output_dim
        base_config['out_filters'] = self.out_filters
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
