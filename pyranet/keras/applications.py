import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import models
from pyranet.keras.layers import WeightedSum3D, MaxPooling3D, ZeroMeanUnitVarianceNormalizer


class StrictPyranet3D(models.Model):

    def __init__(self, num_classes, out_filters, input_shape=None, include_top=True,
                 weight_decay=None, log=None, name="3DStrictPyranet",
                 **kwargs):
        super(StrictPyranet3D, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes

        if input_shape is None:
            input_shape = (100, 100, 1)

        self.ws3d_1 = WeightedSum3D(filters=out_filters, input_shape=input_shape, log=log, name="L1WS")
        self.norm_2 = ZeroMeanUnitVarianceNormalizer(axes=(2, 3), name="NORM_2")

        self.pool3d_3 = MaxPooling3D(weight_decay=weight_decay, log=log, name="L3P")
        self.norm_4 = ZeroMeanUnitVarianceNormalizer(axes=(2, 3), name="NORM_4")

        self.ws3d_5 = WeightedSum3D(filters=out_filters, log=log, name="L5WS")
        self.norm_6 = ZeroMeanUnitVarianceNormalizer(axes=(2, 3), name="NORM_6")

        self.include_top = include_top
        self.logits = None
        if include_top:
            self.logits = keras.layers.Dense(units=num_classes, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.ws3d_1(inputs)
        x = self.norm_2(x)

        x = self.pool3d_3(x)
        x = self.norm_4(x)

        x = self.ws3d_5(x)
        x = self.norm_6(x)

        if self.include_top:
            x = keras.layers.Flatten()(x)
            x = self.logits(x)

        return x
