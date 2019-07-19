import tensorflow as tf
from ..pyranet import *

# Check formula
def ws3d_layer_output_shape_transposed(input_shape, rf=(3, 4, 4), strides=(1, 1, 1, 1, 1), padding="VALID"):
    padding = padding.upper()
    input_shape = list(map(float, input_shape))
    if padding == "VALID":
        output_depth = np.round((input_shape[0] - rf[0] + 1.) / strides[1])
        output_height = np.round((input_shape[1] - rf[1] + 1.) / strides[2])
        output_width = np.round((input_shape[2] - rf[2] + 1.) / strides[3])
        # output_depth = np.round((input_shape[0] - receptive_field[0]) / strides[1] + 1.)
        # output_height = np.round((input_shape[1] - receptive_field[1]) / strides[2] + 1.)
        # output_width = np.round((input_shape[2] - receptive_field[2]) / strides[3] + 1.)
    elif padding == "SAME":
        output_depth, output_height, output_width = [np.round(s / strides[i])
                                                     for i, s in zip(strides[1:-1], input_shape)]
    else:
        raise NotImplementedError("{} is not a valid padding type".format(padding))

    return output_depth, output_height, output_width


def padding_tansposed(in_size, stride, padding, kernel_size):
    padding_t = kernel_size - 0
    output_size = in_size + (kernel_size - 1)


def padding_size(output_size, stride, kernel_size, in_size):
    x = (output_size - 1) * stride + kernel_size - in_size
    x_side = x / 2
    x_side = 0 if x_side < 0 else x_side
    return [x_side, x_side] if x % 2 == 0 or x < 0 else [x_side, x_side + 1]


def _ws3d_same_padding(input_tensor, weights, rf=(3, 4, 4), strides=(1, 1, 1, 1, 1),
                           padding="VALID", data_format="NDHWC", name="ws3d"):
    """
    PyraNetwith padding SAME

    :param input_tensor:
    :param weights:
    :param rf: a list containing the receptive field sizes [height, width], it is the r_l in the original paper
    :param strides: overlap of receptive fields [1, stride_h, stride_w, 1], it is the o_l in the original paper
    :param padding: only VALID is admitted
    :param data_format:
    :param name:
    :return:
    """

    with tf.name_scope(name):

        input_shape = list(input_tensor.shape)
        if data_format == "NDHWC":
            n, d, h, w, c = list(map(int, input_shape))
        else:
            n, c, d, h, w = list(map(int, input_shape))

        out_channels = int(weights.shape[-1])

        output_depth, output_height, output_width = list(map(int,
                                                        ws3d_layer_output_shape_new((d, h, w), rf=rf, strides=strides,
                                                                                    padding=padding)))
        # print output_depth, output_height, output_width
        d_pad_list = padding_size(output_depth, strides[1], rf[0], d)
        h_pad_list = padding_size(output_height, strides[2], rf[1], h)
        w_pad_list = padding_size(output_width, strides[3], rf[2], w)
        # print d_pad_list, h_pad_list, w_pad_list

        if padding.upper() == "SAME":
            input_tensor = tf.pad(input_tensor, paddings=[[0, 0], d_pad_list, [0, 0], [0, 0], [0, 0]])
            # pad also weights? it helps to avoid multiple padding in feature maps loop

        correlation = []

        # Bias and conv need to be intern to weighting
        with tf.name_scope("Input_Weighting_Op"):
            conv_weights = tf.constant(1.0, tf.float32, shape=(rf[0], rf[1], rf[2], c, 1),
                                       name="{}/conv_kernel".format(name))

            for fm in range(out_channels):
                assign_ops = []
                for cd in range(output_depth):
                    s = cd * strides[1]
                    out_mul = tf.multiply(input_tensor[:, s:s + rf[0], :, :, :], weights[:, :, :, :, fm])

                    if padding.upper() == "SAME":
                        out_mul = tf.pad(out_mul, paddings=[[0, 0], [0, 0], h_pad_list, w_pad_list, [0, 0]])

                    with tf.name_scope("Correlation_Op"):
                        corr = tf.nn.conv3d(out_mul, conv_weights,
                                            padding="VALID", strides=strides, name="xcorr3d")
                        # print cd, ":", out_mul, corr
                    assign_ops.append(corr)
                # print tf.identity(assign_ops)

                correlation.append(assign_ops)
            # print tf.identity(correlation)
            corr_axis_sorted = tf.transpose(correlation, [2, 1, 3, 4, 5, 0, 6], name="xcorr_sorted")
            # Shape is like: (10, 3, 1, 5, 5, 9, 1), it is needed to remove 1 from axis 2 and 6
            return tf.squeeze(corr_axis_sorted, axis=[2, 6], name="xcorr_output")


def _ws3d_transposed(input_tensor, weights, rf=(3, 4, 4), strides=(1, 1, 1, 1, 1),
                           padding="VALID", data_format="NDHWC", name="ws3d"):
    """
    PyraNetwith padding SAME

    :param input_tensor:
    :param weights:
    :param rf: a list containing the receptive field sizes [height, width], it is the r_l in the original paper
    :param strides: overlap of receptive fields [1, stride_h, stride_w, 1], it is the o_l in the original paper
    :param padding: only VALID is admitted
    :param data_format:
    :param name:
    :return:
    """

    with tf.name_scope(name):

        input_shape = list(input_tensor.shape)
        if data_format == "NDHWC":
            n, d, h, w, c = list(map(int, input_shape))
        else:
            n, c, d, h, w = list(map(int, input_shape))

        out_channels = int(weights.shape[-1])

        output_depth, output_height, output_width = list(map(int,
                                                             ws3d_layer_output_shape_new((d, h, w), rf=rf,
                                                                                         strides=strides,
                                                                                         padding=padding)))
        # print output_depth, output_height, output_width
        d_pad_list = padding_size(output_depth, strides[1], rf[0], d)
        h_pad_list = padding_size(output_height, strides[2], rf[1], h)
        w_pad_list = padding_size(output_width, strides[3], rf[2], w)
        # print d_pad_list, h_pad_list, w_pad_list

        # if padding.upper() == "SAME":
        input_tensor = tf.pad(input_tensor, paddings=[[0, 0], d_pad_list, [0, 0], [0, 0], [0, 0]])
        # pad also weights? it helps to avoid multiple padding in feature maps loop

        correlation = []

        # Bias and conv need to be intern to weighting
        with tf.name_scope("Input_Weighting_Op"):
            conv_weights = tf.constant(1.0, tf.float32, shape=(rf[0], rf[1], rf[2], c, 1),
                                       name="{}/conv_kernel".format(name))

            for fm in range(out_channels):
                assign_ops = []
                for cd in range(output_depth):
                    s = cd * strides[1]
                    out_mul = tf.multiply(input_tensor[:, s:s + rf[0], :, :, :], weights[:, :, :, :, fm])

                    # if padding.upper() == "SAME":
                    out_mul = tf.pad(out_mul, paddings=[[0, 0], [0, 0], h_pad_list, w_pad_list, [0, 0]])

                    with tf.name_scope("Correlation_Op"):
                        corr = tf.nn.conv3d(out_mul, conv_weights,
                                            padding="VALID", strides=strides, name="xcorr3d")
                        # print cd, ":", out_mul, corr
                    assign_ops.append(corr)
                # print tf.identity(assign_ops)

                correlation.append(assign_ops)
            # print tf.identity(correlation)
            corr_axis_sorted = tf.transpose(correlation, [2, 1, 3, 4, 5, 0, 6], name="xcorr_sorted")
            # Shape is like: (10, 3, 1, 5, 5, 9, 1), it is needed to remove 1 from axis 2 and 6
            return tf.squeeze(corr_axis_sorted, axis=[2, 6], name="xcorr_output")
