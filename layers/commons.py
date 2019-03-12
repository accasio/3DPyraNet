import tensorflow as tf
from models.pyranet import *
from .variables import *


def normalization_layer(x, axes=(2, 3), epsilon=1e-8, name="norm_layer"):
    with tf.name_scope(name):
        beta = tf.constant(0., dtype=tf.float32)
        gamma = tf.constant(1., dtype=tf.float32)
        mean, var = tf.nn.moments(x, axes, keep_dims=True)
        normalized = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon, name="standardization")
        normalized = tf.where(tf.is_finite(normalized), normalized, tf.zeros_like(normalized))  # avoid nan and inf
        return normalized


def fc_layer(inputs, weight_size, name, act_fn=None, weight_decay=None):
    with tf.variable_scope(name):
        if len(inputs.shape) > 2:
            inputs = tf.reshape(inputs, [int(inputs.shape[0]), -1])

        w = get_variable_with_decay("weights", shape=[int(inputs.shape[1]), weight_size],
                                      initializer=tf.contrib.
                                      layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                                          uniform=True, dtype=tf.float32),
                                      weight_decay=weight_decay)
        b = get_variable_with_decay("bias", shape=[weight_size],
                                      initializer=tf.contrib.
                                      layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                                          uniform=True, dtype=tf.float32))

        net = tf.add(tf.matmul(inputs, w), b, name="mat_mul_bias_add")
        if act_fn:
            net = act_fn(net)

        return net
