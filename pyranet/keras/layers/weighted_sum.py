import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import initializers
from tensorflow.python.keras import activations
from pyranet.layers.utils import *
from pyranet.models.pyranet import *


class WeightedSum3D(layers.Layer):

    def __init__(self, filters, receptive_field=(3, 4, 4), strides=(1, 1, 1), activation=activations.relu,
                 kernel_initializer=None, bias_initializer=None, weight_decay=None, padding="VALID",
                 data_format="NDHWC", log=False, name="weighted_sum_3d_layer", **kwargs):
        super(WeightedSum3D, self).__init__(name=name, **kwargs)

        self.filters = filters
        self.activation = activations.get(activation)

        self.receptive_field = check_receptive_field(receptive_field, 3)
        self.strides = check_strides(strides, 3)

        self.kernel_initializer = check_variable_initializer(kernel_initializer, default='XAVIER')
        self.bias_initializer = check_variable_initializer(bias_initializer, default='XAVIER')
        self.weight_decay = weight_decay

        self.padding = padding
        self.data_format = data_format

        self.kernel = None
        self.bias = None

        self.log = log

    def build(self, input_shape):
        if isinstance(input_shape, tf.TensorShape):
            _, d, h, w, c = [x.value for x in input_shape]
        else:
            _, d, h, w, c = input_shape
        kernel_shape = self.receptive_field[0], h, w, c, self.filters
        bias_shape = self.compute_output_shape(input_shape).as_list()[1:]

        self.kernel = self.add_weight(name='kernel',
                                      shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      regularizer=tf.keras.regularizers.l2(self.weight_decay)
                                      if self.weight_decay else None,
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=bias_shape,
                                    initializer=self.kernel_initializer,
                                    regularizer=tf.keras.regularizers.l2(self.weight_decay)
                                    if self.weight_decay else None,
                                    trainable=True)

        # Make sure to call the `build` method at the end
        self.built = True
        super(WeightedSum3D, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = ws3d_base(inputs, self.kernel, rf=self.receptive_field, strides=self.strides,
                      padding=self.padding, data_format=self.data_format, name=self._name)

        x = tf.add(x, self.bias, name="bias_add")

        if self.activation:
            x = self.activation(x)

        if self.log:
            tf.logging.info("{} | RF: {} - strides: {}".format(self._name, self.receptive_field, self.strides[1:-1]))
            tf.logging.info("\t{} {}".format(self.kernel.name, self.kernel.shape))
            tf.logging.info("\t{} {}".format(self.bias.name, self.bias.shape))
            tf.logging.info("\t{} {}".format(x.name, x.shape))

        return x

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, tf.TensorShape):
            input_shape = tuple(x.value for x in input_shape)
        shape = (None,) + ws3d_layer_output_shape(input_shape[1:-1] + (self.filters,), rf=self.receptive_field,
                                                  strides=self.strides, padding=self.padding)
        return tf.TensorShape(shape)

    def get_config(self):
        config = {
            'filters': self.filters,
            'activation': activations.serialize(self.activation),
            'receptive_field': self.receptive_field,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'weight_decay': self.weight_decay,
            'log': self.log
        }

        base_config = super(WeightedSum3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
