import tensorflow as tf
from ..models.pyranet import *
from .variables import *
from .utils import check_variable_initializer


def ws2d(input_data, out_filters, rf=4, strides=1, act_fn=lrelu,
         kernel_initializer=None, bias_initializer=None, weight_decay=None, padding="VALID",
         data_format="NHWC", log=False, reuse=False, name="weighted_sum_2d_layer"):
    """Executes the Weighted Sum 2D layer on the input. It uses the 3D weighted sum to perform the computation
    extending the 2D input dimension in order to match the requirements of 3D layer (the new dimension doesn't
    influence results, it can be considered like a placeholder)

    Args:
        input_data (List[int]): A 4D tensor with dimensions ``[batch_size, height, width, in_channels]``
        out_filters (int): Number of output filters
        rf (Union[int, Tuple(int, int)]): Receptive field (filter mask) size, can be an integer
            (same size will be used on all dimensions) or a tuple indicating (height, width)
        strides (Union[int, Tuple(int, int)]): Sliding step, can be an integer
            (same step will be used on all dimensions) or a tuple indicating (height, width)
        act_fn: A valid activation function handler. Default is provided leaky_relu
        kernel_initializer: Initializer used for kernel weights, default None (uses Xavier initializer)
        bias_initializer: Initializer used for bias, default None (uses Xavier initializer)
        weight_decay: L2 decay lambda value.
        padding (str): Type of padding, only VALID is supported.
        data_format (str): NHWC : Batch x Height x Width x Channels
        log (bool): Log networks structure (weights, bias and output)
        reuse (bool): Not used.
        name (str): Layer name, used in variable_scope

    Returns:
        Weighted Sum 2D tensor with output of size ``[batch_size, out_height, out_width, out_channels]``

    """
    if type(rf) == int:
        rf = (1, rf, rf)
    elif len(rf) == 2:
        rf = (1, ) + (rf, rf)
    else:
        tf.logging.error("Receptive field size in iterable format must be 2")
        raise ValueError("Receptive field size in iterable format must be 2")

    if type(strides) == int:
        strides = (1, 1, strides, strides, 1)
    elif len(strides) < 4 and len(strides) == 2:
        strides = (1, 1) + strides + (1, )
    elif len(strides) == 4:
        if strides[0] != 1 or strides[-1] != 1:
            tf.logging.warn("Strides first and last dimension must be equal to 1. They will set to 1 automatically")
            strides = (1, 1) + strides[1:-2] + (1, )
        else:
            strides = (1, ) + strides

    if data_format == "NHWC":
        data_format = "NDHWC"

    with tf.name_scope(name):
        input_data = tf.expand_dims(input_data, axis=1)
        ws3d_output = ws3d(input_data, out_filters, rf=rf, strides=strides, act_fn=act_fn,
                           kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                           weight_decay=weight_decay, padding=padding, data_format=data_format,
                           log=log, reuse=reuse, name=name)
        return tf.squeeze(ws3d_output, 1)


def ws3d(input_data, out_filters, rf=(3, 4, 4), strides=(1, 1, 1), act_fn=lrelu,
         kernel_initializer=None, bias_initializer=None, weight_decay=None, padding="VALID",
         data_format="NDHWC", log=False, reuse=False, name="weighted_sum_3d_layer"):
    """Executes the Weighted Sum 3D layer on the input

    Args:
        input_data (List[int]): A 5D tensor with dimensions ``[batch_size, depth, height, width, in_channels]``
        out_filters (int): Number of output filters
        rf (Union[int, Tuple(int, int, int)]): Receptive field (filter mask) size, can be an integer
            (same size will be used on all dimensions) or a tuple indicating (depth, height, width)
        strides (Union[int, Tuple(int, int, int)]): Sliding step, can be an integer
            (same step will be used on all dimensions) or a tuple indicating (depth, height, width)
        act_fn: A valid activation function handler. Default is provided leaky_relu
        kernel_initializer: Initializer used for kernel weights, default None (uses Xavier initializer)
        bias_initializer: Initializer used for bias, default None (uses Xavier initializer)
        weight_decay: L2 decay lambda value.
        padding (str): Type of padding, only VALID is supported.
        data_format (str): NDHWC : Batch x Depth x Height x Width x Channels
        log (bool): Log networks structure (weights, bias and output)
        reuse (bool): Not used.
        name (str): Layer name, used in variable_scope

    Returns:
        Weighted Sum 3D tensor with output of size ``[batch_size, out_depth, out_height, out_width, out_channels]``

    """

    if type(rf) == int:
        rf = (rf, rf, rf)
    elif len(rf) > 3:
        tf.logging.error("Receptive field size in iterable format must be 3")
        raise ValueError("Receptive field size in iterable format must be 3")

    if type(strides) == int:
        strides = (1, strides, strides, strides, 1)
    elif len(strides) < 5 and len(strides) == 3:
        strides = (1, ) + strides + (1, )
    elif len(strides) == 5 and (strides[0] != 1 or strides[-1] != 1):
        tf.logging.warn("Strides first and last dimension must be equal to 1. They will set to 1 automatically")
        strides = (1, ) + strides[1:-2] + (1, )

    with tf.variable_scope(name, reuse=reuse):
        _, d, h, w, c = map(int, input_data.shape)
        kernel_initializer = check_variable_initializer(kernel_initializer, default='Xavier')
        bias_initializer = check_variable_initializer(bias_initializer, default='Xavier')

        weights = ws3d_weight_initializer("weights", shape=(rf[0], h, w, c, out_filters),
                                          initializer=kernel_initializer, weight_decay=weight_decay)

        net = ws3d_base(input_data, weights, rf=rf, strides=strides,
                   padding=padding, data_format=data_format, name="ws3d")

        bias = ws3d_bias_initializer_like("bias", tensor=net, initializer=bias_initializer)

        net = tf.add(net, bias, name="bias_add")

        if act_fn:
            net = act_fn(net)

        if log:
            tf.logging.info("{} | RF: {} - strides: {}".format(name, rf, strides[1:-1]))
            tf.logging.info("\t{} {}".format(weights.name, weights.shape))
            tf.logging.info("\t{} {}".format(bias.name, bias.shape))
            tf.logging.info("\t{} {}".format(net.name, net.shape))

        return net
