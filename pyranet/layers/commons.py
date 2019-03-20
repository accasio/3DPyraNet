import tensorflow as tf
from ..models.pyranet import *
from .variables import *
from .utils import check_variable_initializer


def normalization_2d(x, axes=(1, 2), epsilon=1e-8, name="norm_layer_2d"):
    """Apply a zero mean - unit variance normalization (standardization)

    Args:
        x: 4D Tensor to normalize. It expects a data in format `NHWC`.
        axes: Axes where apply the normalization. Default: (1, 2) - [Height, Width].
        epsilon: A small floating point number to avoid dividing by zero.
        name: Operation name.

    Returns: Normalized tensor

    """
    return normalization(x, axes=axes, epsilon=epsilon, name=name)


def normalization_3d(x, axes=(2, 3), epsilon=1e-8, name="norm_layer_3d"):
    """Apply a zero mean - unit variance normalization (standardization)

        Args:
            x: 5D Tensor to normalize. It expects a data in format `NDHWC`.
            axes: Axes where apply the normalization. Default: (2, 3) - [Height, Width].
            epsilon: A small floating point number to avoid dividing by zero.
            name: Operation name.

        Returns: Normalized tensor

        """
    return normalization(x, axes=axes, epsilon=epsilon, name=name)


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


def fully_connected(inputs, features_out, name, weights_initializer=None, bias_initializer=None,
                    act_fn=None, weight_decay=None):
    """
    Fully Connected Layer (or Dense Layer). It automatically reshapes a tensor to get a 2D input preserving batch size
    on first dimension [Batch x Features].

    Args:
        inputs: Input tensor of any size.
        features_out: Number of output features, or number of classes if used as output layer.
        name: Operation name.
        weights_initializer: Weights initializer function. Default None uses XAVIER.
        bias_initializer: Bias initializer function. Default None uses XAVIER.
        act_fn: Activation function handler.
        weight_decay: L2 decay lambda value.

    Returns: Tensor of shape ``[Batch x Features_Out]``

    """
    with tf.variable_scope(name):
        if len(inputs.shape) > 2:
            inputs = tf.reshape(inputs, [int(inputs.shape[0]), -1])

        weights_initializer = check_variable_initializer(weights_initializer, 'XAVIER')
        bias_initializer = check_variable_initializer(bias_initializer, 'XAVIER')

        w = get_variable_with_decay("weights", shape=[int(inputs.shape[1]), features_out],
                                    initializer=weights_initializer,
                                    weight_decay=weight_decay)

        b = get_variable_with_decay("bias", shape=[features_out],
                                    initializer=bias_initializer)

        net = tf.add(tf.matmul(inputs, w), b, name="mat_mul_bias_add")
        if act_fn:
            net = act_fn(net)

        return net
