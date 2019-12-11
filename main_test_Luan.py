import os
import scipy.misc
import numpy as np
import time

from glob import glob

#from utils import *

#import tensorflow as tf
from ops import *

from mobilenet.mobilenet_v2_FR import mobilenet_v2_FR_sz224


def discriminator(image,  is_reuse=False, is_training = True):
    df_dim = 32
    d_bn0_0 = batch_norm(name='d_k_bn0_0')
    d_bn0_1 = batch_norm(name='d_k_bn0_1')
    d_bn0_2 = batch_norm(name='d_k_bn0_2')
    d_bn1_0 = batch_norm(name='d_k_bn1_0')
    d_bn1_1 = batch_norm(name='d_k_bn1_1')
    d_bn1_2 = batch_norm(name='d_k_bn1_2')
    d_bn1_3 = batch_norm(name='d_k_bn1_3')
    d_bn2_0 = batch_norm(name='d_k_bn2_0')
    d_bn2_1 = batch_norm(name='d_k_bn2_1')
    d_bn2_2 = batch_norm(name='d_k_bn2_2')
    d_bn3_0 = batch_norm(name='d_k_bn3_0')
    d_bn3_1 = batch_norm(name='d_k_bn3_1')
    d_bn3_2 = batch_norm(name='d_k_bn3_2')
    d_bn3_3 = batch_norm(name='d_k_bn3_3')
    d_bn4_0 = batch_norm(name='d_k_bn4_0')
    d_bn4_1 = batch_norm(name='d_k_bn4_1')
    d_bn4_2 = batch_norm(name='d_k_bn4_2')
    d_bn5   = batch_norm(name='d_k_bn5')
    

    s16 = int(96/16)
    k0_0 = image
    k0_1 = elu(d_bn0_1(conv2d(k0_0, df_dim*1, d_h=1, d_w =1, name='d_k01_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k01_prelu')
    k0_2 = elu(d_bn0_2(conv2d(k0_1, df_dim*2, d_h=1, d_w =1, name='d_k02_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k02_prelu')
    k1_0 =               maxpool2d(k0_2, k=2, padding='VALID')
    #k1_0 = elu(d_bn1_0(conv2d(k0_2, df_dim*2, d_h=2, d_w =2, name='d_k10_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k10_prelu')
    k1_1 = elu(d_bn1_1(conv2d(k1_0, df_dim*2, d_h=1, d_w =1, name='d_k11_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k11_prelu')
    k1_2 = elu(d_bn1_2(conv2d(k1_1, df_dim*4, d_h=1, d_w =1, name='d_k12_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k12_prelu')
    k2_0 =               maxpool2d(k1_2, k=2, padding='VALID')
    #k2_0 = elu(d_bn2_0(conv2d(k1_2, df_dim*4, d_h=2, d_w =2, name='d_k20_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k20_prelu')
    k2_1 = elu(d_bn2_1(conv2d(k2_0, df_dim*3, d_h=1, d_w =1, name='d_k21_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k21_prelu')
    k2_2 = elu(d_bn2_2(conv2d(k2_1, df_dim*6, d_h=1, d_w =1, name='d_k22_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k22_prelu')
    k3_0 =               maxpool2d(k2_2, k=2, padding='VALID')
    #k3_0 = elu(d_bn3_0(conv2d(k2_2, df_dim*6, d_h=2, d_w =2, name='d_k30_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k30_prelu')
    k3_1 = elu(d_bn3_1(conv2d(k3_0, df_dim*4, d_h=1, d_w =1, name='d_k31_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k31_prelu')
    k3_2 = elu(d_bn3_2(conv2d(k3_1, df_dim*8, d_h=1, d_w =1, name='d_k32_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k32_prelu')
    k4_0 =               maxpool2d(k3_2, k=2, padding='VALID')
    #k4_0 = elu(d_bn4_0(conv2d(k3_2, df_dim*8, d_h=2, d_w =2, name='d_k40_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k40_prelu')
    k4_1 = elu(d_bn4_1(conv2d(k4_0, df_dim*5, d_h=1, d_w =1, name='d_k41_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k41_prelu')
    k4_2 =     d_bn4_2(conv2d(k4_1, 320,  d_h=1, d_w =1, name='d_k42_conv', reuse = is_reuse), train=is_training, reuse = is_reuse)

    k5 = tf.nn.avg_pool(k4_2, ksize = [1, s16, s16, 1], strides = [1,1,1,1],padding = 'VALID')
    k5 = tf.reshape(k5, [-1, 320])
    #if (is_training):
    #    k5 = tf.nn.dropout(k5, keep_prob = 0.6)

    #k6_real = linear(k5, 1,                'd_k6_real_lin')
    k6_id   = linear(k5, 1001, 'd_k6_id_lin')
    #k6_pose = linear(k5, pose_dim,    'd_k6_pose_lin')

    return k6_id, k5 #tf.nn.sigmoid(k6_real), k6_real, tf.nn.softmax(k6_id), k6_id, tf.nn.softmax(k6_pose), k6_pose, k5 


def main(_):

    gpu_options = tf.GPUOptions(visible_device_list ="0", allow_growth = True)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)) as sess:
        #logits, end_points = mobilenet_v2_FR_sz224(tf.random_normal(shape=[2, 96, 96, 3]), is_reuse=False, is_training = False)
        _, logits = discriminator(tf.random_normal(shape=[2, 96, 96, 3]), is_reuse=False, is_training = False)

        t_vars = tf.trainable_variables()
        for var in t_vars:
            print(var.name)
            print(var.shape)

        tf.global_variables_initializer().run()

        #print(end_points.keys())
        #print(end_points['global_pool'].get_shape())

        print(logits.get_shape())

        startTime = time.time()
        for _ in range(100):
            sess.run(logits)
        print(time.time() - startTime)

        startTime = time.time()
        for _ in range(100):
            sess.run(logits)
        print(time.time() - startTime)

        startTime = time.time()
        for _ in range(100):
            sess.run(logits)
        print(time.time() - startTime)




if __name__ == '__main__':
    tf.app.run()
