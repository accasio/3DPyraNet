import tensorflow as tf
from tensorflow.python.keras import layers


class ZeroMeanUnitVarianceNormalizer(layers.Layer):

    def __init__(self, axes, epsilon=1e-8, name="norm_layer", **kwargs):
        super(ZeroMeanUnitVarianceNormalizer, self).__init__(name=name, **kwargs)
        self.axes = axes
        self.epsilon = epsilon

    def build(self, input_shape):
        self.built = True
        super(ZeroMeanUnitVarianceNormalizer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        beta = tf.constant(0., dtype=tf.float32)
        gamma = tf.constant(1., dtype=tf.float32)
        mean, var = tf.nn.moments(inputs, self.axes, keep_dims=True)
        normalized = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, self.epsilon, name="standardization")
        normalized = tf.where(tf.is_finite(normalized), normalized, tf.zeros_like(normalized))  # avoid nan and inf
        return normalized

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'axes': self.axes,
            'epsilon': self.epsilon
        }
        base_config = super(ZeroMeanUnitVarianceNormalizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def normalization(x, axes, epsilon=1e-8, name="norm_layer"):
    """Apply a zero mean - unit variance normalization (standardization)
    implicitly uses batch normalization operation that wraps common standardization

        Args:
            x: ND Tensor to normalize.
            axes: Axes where apply the normalization.
            epsilon: A small floating point number to avoid dividing by zero.
            name: Operation name.

        Returns: Normalized tensor

        """
    with tf.name_scope(name):
        beta = tf.constant(0., dtype=tf.float32)
        gamma = tf.constant(1., dtype=tf.float32)
        mean, var = tf.nn.moments(x, axes, keep_dims=True)
        normalized = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon, name="standardization")
        normalized = tf.where(tf.is_finite(normalized), normalized, tf.zeros_like(normalized))  # avoid nan and inf
        return normalized