import tensorflow as tf
import numpy as np
from pyranet.datasets import input_data
from pyranet.keras import layers, applications
import os
import time
from tqdm import trange
import sys

flags = tf.app.flags

# Checkpoint settings
flags.DEFINE_float("evaluate_every", 1, "Number of epoch for each evaluation (decimals allowed)")
flags.DEFINE_string("test_milestones", "15,20,25,30,35,40,45,50,75,100", "Each epoch where performs test")
flags.DEFINE_boolean("save_checkpoint", False, "Flag to save checkpoint or not")
flags.DEFINE_string("checkpoint_name", "3dpyranet.ckpt", "Name of checkpoint file")

# Input settings
dataset_path = "path/to/dataset"
flags.DEFINE_string("train_path",
                    os.path.join(dataset_path, "Training.npy"),
                    "Path to npy training set")
flags.DEFINE_string("train_labels_path",
                    os.path.join(dataset_path, "Training_label.npy"),
                    "Path to npy training set labels")
flags.DEFINE_string("val_path",
                    os.path.join(dataset_path, "TestVal.npy"),
                    "Path to npy val/test set")
flags.DEFINE_string("val_labels_path",
                    os.path.join(dataset_path, "TestVal_label.npy"),
                    "Path to npy val/test set labels")
flags.DEFINE_string("save_path", "train_dir",
                    "Path where to save network model")
flags.DEFINE_boolean("random_run", False, "Set usage of random data for debug purpose")

# Input parameters
flags.DEFINE_integer("batch_size", 100, "Batch size")
flags.DEFINE_integer("depth", 16, "Number of consecutive samples")
flags.DEFINE_integer("height", 100, "Samples height")
flags.DEFINE_integer("width", 100, "Samples width")
flags.DEFINE_integer("in_channels", 1, "Samples channels")
flags.DEFINE_integer("num_classes", 3, "Number of classes")

# Preprocessing
flags.DEFINE_boolean("normalize", True, "Normalize image in range 0-1")

# Hyper-parameters settings
flags.DEFINE_integer("feature_maps", 3, "Number of maps to use (strict model shares the number of maps in each layer)")
flags.DEFINE_float("learning_rate", 0.00015, "Learning rate")
flags.DEFINE_integer("decay_steps", 15, "Number of iteration for each decay")
flags.DEFINE_float("decay_rate", 0.1, "Learning rate decay")
flags.DEFINE_integer("max_steps", 50, "Maximum number of epoch to perform")
flags.DEFINE_float("weight_decay", None, "L2 regularization lambda")

# Optimization algorithm
opt_type = ["GD", "MOMENTUM", "ADAM"]
flags.DEFINE_string("optimizer", opt_type[1], "Optimization algorithm")
flags.DEFINE_boolean("use_nesterov", False, "Use Nesterov Momentum")

params_str = ""
FLAGS = flags.FLAGS
FLAGS(sys.argv)

print("Parameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    params_str += "{} = {}\n".format(attr.upper(), value)
    print("{} = {}".format(attr.upper(), value.value))
print("")


def compute_loss(name_scope, logits, labels):
    with tf.name_scope("Loss_{}".format(name_scope)):
        cross_entropy_mean = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        )

        tf.summary.scalar(
            name_scope + '_cross_entropy',
            cross_entropy_mean
        )

        weight_decay_loss = tf.get_collection('weight_decay')

        if len(weight_decay_loss) > 0:
            tf.summary.scalar(name_scope + '_weight_decay_loss', tf.reduce_mean(weight_decay_loss))

            # Calculate the total loss for the current tower.
            total_loss = cross_entropy_mean + weight_decay_loss
            tf.summary.scalar(name_scope + '_total_loss', tf.reduce_mean(total_loss))
        else:
            total_loss = cross_entropy_mean

        return total_loss


def compute_accuracy(logits, labels):
    with tf.name_scope("Accuracy"):
        correct_pred = tf.equal(tf.argmax(logits, 1), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar("accuracy", accuracy)
        return accuracy


def prepare_dataset():
    train_x, train_y, val_x, val_y = input_data.read_dataset(
        FLAGS.train_path,
        FLAGS.train_labels_path,
        FLAGS.val_path,
        FLAGS.val_labels_path
    )

    batch_step = train_x.shape[0] // FLAGS.batch_size
    test_batch_step = val_x.shape[0] // FLAGS.batch_size

    FLAGS.decay_steps *= batch_step
    FLAGS.max_steps *= batch_step
    FLAGS.evaluate_every = int(FLAGS.evaluate_every * batch_step)

    if FLAGS.normalize:
        train_x = input_data.normalize(train_x, name="training set")
        val_x = input_data.normalize(val_x, name="val set")

    train_batch = input_data.generate_batch(train_x, train_y, batch_size=FLAGS.batch_size, shuffle=True)
    val_batch = input_data.generate_batch(val_x, val_y, batch_size=FLAGS.batch_size, shuffle=True)
    test_batch = input_data.generate_batch(val_x, val_y, batch_size=FLAGS.batch_size, shuffle=False)

    return batch_step, test_batch_step, train_batch, val_batch, test_batch


def random_dataset():
    train_x = np.random.rand(FLAGS.batch_size, FLAGS.depth, FLAGS.height, FLAGS.width, FLAGS.in_channels)
    train_y = np.random.randint(0, FLAGS.num_classes, size=FLAGS.batch_size)
    val_x = np.random.rand(FLAGS.batch_size, FLAGS.depth, FLAGS.height, FLAGS.width, FLAGS.in_channels)
    val_y = np.random.randint(0, FLAGS.num_classes, size=FLAGS.batch_size)

    batch_step = train_x.shape[0] // FLAGS.batch_size
    test_batch_step = val_x.shape[0] // FLAGS.batch_size

    FLAGS.decay_steps *= batch_step
    FLAGS.max_steps *= batch_step
    FLAGS.evaluate_every = int(FLAGS.evaluate_every * batch_step)

    if FLAGS.normalize:
        train_x = input_data.normalize(train_x, name="training set")
        val_x = input_data.normalize(val_x, name="val set")

    train_batch = input_data.generate_batch(train_x, train_y, batch_size=FLAGS.batch_size, shuffle=True)
    val_batch = input_data.generate_batch(val_x, val_y, batch_size=FLAGS.batch_size, shuffle=True)
    test_batch = input_data.generate_batch(val_x, val_y, batch_size=FLAGS.batch_size, shuffle=False)

    return batch_step, test_batch_step, train_batch, val_batch, test_batch


def random_dataset_no_batch():
    train_x = np.random.rand(FLAGS.batch_size, FLAGS.depth, FLAGS.height, FLAGS.width, FLAGS.in_channels)
    train_y = np.random.randint(0, FLAGS.num_classes, size=FLAGS.batch_size)
    val_x = np.random.rand(FLAGS.batch_size, FLAGS.depth, FLAGS.height, FLAGS.width, FLAGS.in_channels)
    val_y = np.random.randint(0, FLAGS.num_classes, size=FLAGS.batch_size)

    if FLAGS.normalize:
        train_x = input_data.normalize(train_x, name="training set")
        val_x = input_data.normalize(val_x, name="val set")

    return train_x, train_y, val_x, val_y


def train():
    # if FLAGS.random_run:
    train_x, train_y, val_x, val_y = random_dataset_no_batch()

    def exp_decay(epoch):
        initial_lrate = FLAGS.decay_rate
        k = 0.1
        lrate = initial_lrate * np.exp(-k * epoch)
        return float(lrate)

    lrate_decay = tf.keras.callbacks.LearningRateScheduler(exp_decay)

    optimizer = None
    if FLAGS.optimizer == "GD":
        optimizer = tf.keras.optimizers.SGD(lr=FLAGS.learning_rate, decay=0.0)
    elif FLAGS.optimizer == "ADAM":
        optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    elif FLAGS.optimizer == "MOMENTUM":
        optimizer = tf.keras.optimizers.SGD(lr=FLAGS.learning_rate, decay=0.0,
                                            momentum=0.9, nesterov=FLAGS.use_nesterov)

    # TODO: check model serialization
    # model = applications.StrictPyranet3D(num_classes=FLAGS.num_classes, out_filters=FLAGS.feature_maps,
    #                                      include_top=True, input_shape=(16, 100, 100, 1))

    # or

    model = tf.keras.models.Sequential()
    model.add(layers.WeightedSum3D(filters=FLAGS.feature_maps, input_shape=train_x.shape[1:]))
    model.add(layers.MaxPooling3D())
    # model.add(layers.WeightedSum3D(filters=FLAGS.feature_maps))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(FLAGS.num_classes, activation='softmax'))
    model.summary()

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['sparse_categorical_accuracy'])

    model.fit(train_x, train_y, batch_size=FLAGS.batch_size, epochs=FLAGS.max_steps, callbacks=[lrate_decay])


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
