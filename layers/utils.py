import tensorflow as tf


def check_initializer(initializer_type, default):
    if initializer_type is None:
        if default == 'XAVIER':
            return tf.contrib.ayers.\
                variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                             uniform=True, dtype=tf.float32)
        else:
            return tf.initializers.truncated_normal()
    else:
        return initializer_type


def check_receptive_field(rf, size):
    if size == 2:
        if type(rf) == int:
            rf = 1, rf, rf
        elif len(rf) == 2:
            rf = (1, ) + (rf, rf)
        else:
            tf.logging.error("Receptive field size in iterable format must be 2")
            raise ValueError("Receptive field size in iterable format must be 2")
    else:
        if type(rf) == int:
            rf = rf, rf, rf
        elif len(rf) > 3:
            tf.logging.error("Receptive field size in iterable format must be 3")
            raise ValueError("Receptive field size in iterable format must be 3")
    return rf


def check_strides(strides, size):
    if size == 2:
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
    else:
        if type(strides) == int:
            strides = (1, strides, strides, strides, 1)
        elif len(strides) < 5 and len(strides) == 3:
            strides = (1,) + strides + (1,)
        elif len(strides) == 5 and (strides[0] != 1 or strides[-1] != 1):
            tf.logging.warn("Strides first and last dimension must be equal to 1. They will set to 1 automatically")
            strides = (1,) + strides[1:-2] + (1,)
    return strides
