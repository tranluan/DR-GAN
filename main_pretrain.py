import os
import scipy.misc
import numpy as np

from model_pretrain import DCGAN
from utils import pp, visualize, to_json

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 1000, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", 10000000, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("sample_size", 169, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 96, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_boolean("is_with_y", True, "True for with lable")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("samples_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("gf_dim", 32, "")
flags.DEFINE_integer("gfc_dim", 512, "")
flags.DEFINE_integer("df_dim", 32, "")
flags.DEFINE_integer("dfc_dim", 512, "")
flags.DEFINE_integer("z_dim", 50, "")
flags.DEFINE_string("gpu", "0,1,2", "GPU to use [0]")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.samples_dir):
        os.makedirs(FLAGS.samples_dir)

    #gpu_options = tf.GPUOptions(visible_device_list =FLAGS.gpu, per_process_gpu_memory_fraction = 0.95, allow_growth = True)

    #with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,  gpu_options=gpu_options)) as sess:
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):
        dcgan = DCGAN(FLAGS)
            
        if FLAGS.is_train:
            dcgan.train(FLAGS)
        else:
            dcgan.load(FLAGS.checkpoint_dir)
            dcgan.test(FLAGS, True)

if __name__ == '__main__':
    tf.app.run()

