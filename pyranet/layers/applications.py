import tensorflow as tf
from .weighted_sum import ws3d
from .commons import normalization_3d
from .pooling import max_pooling3d
from ..models import pyranet


def strict_norm_net(inputs, feature_maps=3, act_fn=pyranet.lrelu, weight_decay=None,
                    log=False, name="STRICT_3DPYRANET"):
    with tf.name_scope(name):
        net = ws3d(inputs, feature_maps, name="L1WS", act_fn=act_fn, weight_decay=weight_decay, log=log)
        net = normalization_3d(net, axes=(2, 3), name="NORM_2")

        net = max_pooling3d(net, name="L3P", act_fn=act_fn, weight_decay=weight_decay, log=log)
        net = normalization_3d(net, axes=(2, 3), name="NORM_4")

        net = ws3d(net, feature_maps, name="L5WS", act_fn=act_fn, weight_decay=weight_decay, log=True)
        net = normalization_3d(net, axes=(2, 3), name="NORM_6")

        return net
