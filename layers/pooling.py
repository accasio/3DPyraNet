from ..models import pyranet
from .variables import *
from .utils import check_initializer, check_receptive_field, check_strides


def max_pooling2d(input_data, rf=(2, 2), strides=(2, 2), act_fn=pyranet.lrelu,
                  kernel_initializer=None, bias_initializer=None, weight_decay=None,
                  padding="VALID", data_format="NHWC", reuse=False, log=False, name="max_pooling_2d_layer"):
    """Executes Pyranet modified Max Pooling 2D on the input. It uses the 3D Max Pooling to perform the computation
    extending the 2D input dimension in order to match the requirements of 3D layer (the new dimension doesn't
    influence results, it can be considered like a placeholder)

    Args:
        input_data (List[int]): A 4D tensor with dimensions ``[batch_size, in_height, in_width, channels]``
        rf (Union[int, Tuple(int, int)]): Receptive field (filter mask) size, can be an integer
            (same size will be used on all dimensions) or a tuple indicating (height, width)
        strides (Union[int, Tuple(int, int)]): Sliding step, can be an integer
            (same step will be used on all dimensions) or a tuple indicating (height, width)
        act_fn: A valid activation function handler. Default is provided leaky_relu
        kernel_initializer: Initializer used for kernel weights, default None (uses Xavier initializer)
        bias_initializer: Initializer used for bias, default None (uses Xavier initializer)
        weight_decay: Weight decay handler function, default None.
        padding (str): Type of padding, only VALID is supported.
        data_format (str): NHWC : Batch x Height x Width x Channels
        log (bool): Log networks structure (weights, bias and output)
        reuse (bool): Not used.
        name (str): Layer name, used in variable_scope

    Returns:
        Max pooled 2D tensor with output of size ``[batch_size, out_height, out_width, channels]``
    """

    rf = check_receptive_field(rf, size=2)
    strides = check_strides(strides, size=2)

    if data_format == "NHWC":
        data_format = "NDHWC"

    with tf.name_scope(name):
        input_data = tf.expand_dims(input_data, axis=1)
        pool3d_output = max_pooling3d(input_data, rf=rf, strides=strides, act_fn=act_fn,
                                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                      weight_decay=weight_decay, padding=padding, data_format=data_format,
                                      log=log, reuse=reuse, name=name)
        return tf.squeeze(pool3d_output, 1)


def avg_pooling2d(input_data, rf=(2, 2), strides=(2, 2), act_fn=pyranet.lrelu,
                  kernel_initializer=None, bias_initializer=None, weight_decay=None,
                  padding="VALID", data_format="NHWC", reuse=False, log=False, name="max_pooling_2d_layer"):
    """Executes Pyranet modified Average Pooling 2D on the input. It uses the 3D Average Pooling to perform
    the computation extending the 2D input dimension in order to match the requirements of 3D layer (the new dimension doesn't
    influence results, it can be considered like a placeholder)

    Args:
        input_data (List[int]): A 4D tensor with dimensions ``[batch_size, in_height, in_width, channels]``
        rf (Union[int, Tuple(int, int)]): Receptive field (filter mask) size, can be an integer
            (same size will be used on all dimensions) or a tuple indicating (height, width)
        strides (Union[int, Tuple(int, int)]): Sliding step, can be an integer
            (same step will be used on all dimensions) or a tuple indicating (height, width)
        act_fn: A valid activation function handler. Default is provided leaky_relu
        kernel_initializer: Initializer used for kernel weights, default None (uses Xavier initializer)
        bias_initializer: Initializer used for bias, default None (uses Xavier initializer)
        weight_decay: Weight decay handler function, default None.
        padding (str): Type of padding, only VALID is supported.
        data_format (str): NHWC : Batch x Height x Width x Channels
        log (bool): Log networks structure (weights, bias and output)
        reuse (bool): Not used.
        name (str): Layer name, used in variable_scope

    Returns:
        Max pooled 2D tensor with output of size ``[batch_size, out_height, out_width, channels]``
    """

    rf = check_receptive_field(rf, size=2)
    strides = check_strides(strides, size=2)

    if data_format == "NHWC":
        data_format = "NDHWC"

    with tf.name_scope(name):
        input_data = tf.expand_dims(input_data, axis=1)
        pool3d_output = avg_pooling3d(input_data, rf=rf, strides=strides, act_fn=act_fn,
                                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                      weight_decay=weight_decay, padding=padding, data_format=data_format,
                                      log=log, reuse=reuse, name=name)
        return tf.squeeze(pool3d_output, 1)


def max_pooling3d(input_data, rf=(3, 2, 2), strides=(1, 1, 2, 2, 1), act_fn=pyranet.lrelu,
                  kernel_initializer=None, bias_initializer=None, weight_decay=None,
                  padding="VALID", data_format="NDHWC", reuse=False, log=False, name="max_pooling_3d_layer"):
    """Executes Pyranet modified Max Pooling 3D on the input.

    Args:
        input_data (List[int]): A 5D tensor with dimensions
            ``[batch_size, in_depth, in_height, in_width, channels]``
        rf (Union[int, Tuple(int, int, int)]): Receptive field (filter mask) size, can be an integer
            (same size will be used on all dimensions) or a tuple indicating (depth, height, width)
        strides (Union[int, Tuple(int, int, int)]): Sliding step, can be an integer
            (same step will be used on all dimensions) or a tuple indicating (depth, height, width)
        act_fn: A valid activation function handler. Default is provided leaky_relu
        kernel_initializer: Initializer used for kernel weights, default None (uses Xavier initializer)
        bias_initializer: Initializer used for bias, default None (uses Xavier initializer)
        weight_decay: Weight decay handler function, default None.
        padding (str): Type of padding, only VALID is supported.
        data_format (str): NDHWC : Batch x Depth x Height x Width x Channels
        log (bool): Log networks structure (weights, bias and output)
        reuse (bool): Not used.
        name (str): Layer name, used in variable_scope

    Returns:
        Max pooled 3D tensor with output of size ``[batch_size, out_depth, out_height, out_width, channels]``
    """

    with tf.variable_scope(name):
        return pool3d(input_data, rf=rf, strides=strides, act_fn=act_fn,
                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      weight_decay=weight_decay, padding=padding, data_format=data_format,
                      pool_type=pyranet.max_pool3d, log=log, reuse=reuse, name=name)


def avg_pooling3d(input_data, rf=(3, 2, 2), strides=(1, 1, 2, 2, 1), act_fn=pyranet.lrelu,
                  kernel_initializer=None, bias_initializer=None, weight_decay=None,
                  padding="VALID", data_format="NDHWC", reuse=False, log=False, name="max_pooling_3d_layer"):
    """Executes Pyranet modified Average Pooling 3D on the input.

    Args:
        input_data (List[int]): A 5D tensor with dimensions
            ``[batch_size, in_depth, in_height, in_width, channels]``
        rf (Union[int, Tuple(int, int, int)]): Receptive field (filter mask) size, can be an integer
            (same size will be used on all dimensions) or a tuple indicating (depth, height, width)
        strides (Union[int, Tuple(int, int, int)]): Sliding step, can be an integer
            (same step will be used on all dimensions) or a tuple indicating (depth, height, width)
        act_fn: A valid activation function handler. Default is provided leaky_relu
        kernel_initializer: Initializer used for kernel weights, default None (uses Xavier initializer)
        bias_initializer: Initializer used for bias, default None (uses Xavier initializer)
        weight_decay: Weight decay handler function, default None.
        padding (str): Type of padding, only VALID is supported.
        data_format (str): NDHWC : Batch x Depth x Height x Width x Channels
        log (bool): Log networks structure (weights, bias and output)
        reuse (bool): Not used.
        name (str): Layer name, used in variable_scope

    Returns:
        Average pooled 3D tensor with output of size ``[batch_size, out_depth, out_height, out_width, channels]``
    """

    with tf.variable_scope(name):
        return pool3d(input_data, rf=rf, strides=strides, act_fn=act_fn,
                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      weight_decay=weight_decay, padding=padding, data_format=data_format,
                      pool_type=pyranet.avg_pool3d, log=log, reuse=reuse, name=name)


def pool3d(input_data, pool_type, rf=(3, 2, 2), strides=(1, 1, 2, 2, 1), act_fn=pyranet.lrelu,
           kernel_initializer=None, bias_initializer=None, weight_decay=None,
           padding="VALID", data_format="NDHWC", reuse=False, log=False, name="pooling_3d_layer"):
    """Executes Pyranet modified Average Pooling 3D on the input.

    Args:
        input_data (List[int]): A 5D tensor with dimensions
            ``[batch_size, in_depth, in_height, in_width, channels]``
        pool_type: A valid activation function handler (see :models.pyranet:`Pooling`). Default is provided leaky_relu
        rf (Union[int, Tuple(int, int, int)]): Receptive field (filter mask) size, can be an integer
            (same size will be used on all dimensions) or a tuple indicating (depth, height, width)
        strides (Union[int, Tuple(int, int, int)]): Sliding step, can be an integer
            (same step will be used on all dimensions) or a tuple indicating (depth, height, width)
        act_fn: A valid activation function handler. Default is provided leaky_relu
        kernel_initializer: Initializer used for kernel weights, default None (uses Xavier initializer)
        bias_initializer: Initializer used for bias, default None (uses Xavier initializer)
        weight_decay: Weight decay handler function, default None.

        padding (str): Type of padding, only VALID is supported.
        data_format (str): NDHWC : Batch x Depth x Height x Width x Channels
        log (bool): Log networks structure (weights, bias and output)
        reuse (bool): Not used.
        name (str): Layer name, used in variable_scope

    Returns:
        Pooled 3D tensor with output of size ``[batch_size, out_depth, out_height, out_width, channels]``
    """

    rf = check_receptive_field(rf, size=3)
    strides = check_strides(strides, size=3)

    with tf.variable_scope(name, reuse=reuse):
        kernel_initializer = check_initializer(kernel_initializer, default='XAVIER')
        bias_initializer = check_initializer(bias_initializer, default='XAVIER')

        net = pool_type(input_data, rf=rf, strides=strides,
                        padding=padding, data_format=data_format)

        weights = pool3d_weight_initializer_like("weights", tensor=net,
                                                 initializer=kernel_initializer, weight_decay=weight_decay)
        bias = pool3d_bias_initializer_like("bias", tensor=net, initializer=bias_initializer)

        net = tf.multiply(net, weights, name="mul_weights")
        net = tf.add(net, bias, name="bias_add")

        if act_fn:
            net = act_fn(net)

        if log:
            tf.logging.info("{} | Weight depth: {} - strides: {}".format(name, rf[0], strides[1:-1]))
            tf.logging.info("\t{} {}".format(weights.name, weights.shape))
            tf.logging.info("\t{} {}".format(bias.name, bias.shape))
            tf.logging.info("\t{} {}".format(net.name, net.shape))

        return net
