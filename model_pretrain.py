from __future__ import division
import os
import time
import csv
import random
from random import randint
from math import floor
from glob import glob
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from six.moves import xrange
#from progress.bar import Bar
from ops import *
from utils import *
from Loss_AMSoftmax import *
from mobilenet.mobilenet_v2_FR import *

SUBJECT_NUM_VGG2 = 8631



class DCGAN(object):
    def __init__(self, config):
        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [64]
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        #self.sess = sess
        self.is_crop = config.is_crop
        self.is_grayscale = (config.c_dim == 1)
        self.batch_size = config.batch_size
        self.image_size = config.image_size
        self.sample_size = config.sample_size
        self.output_size = config.image_size
        self.gf_dim = config.gf_dim
        self.df_dim = config.df_dim
        self.gfc_dim = config.gfc_dim
        self.dfc_dim = config.dfc_dim
        self.c_dim = config.c_dim
        self.num_gpus = len(config.gpu.split(','))

        self.random_flip = True
        self.padding = 0.15
        self.random_rotate = True
        self.learning_rate = config.learning_rate
        self.beta1 = config.beta1

        self.before_crop_size = int(self.output_size * (1 + self.padding))
        
        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn0_0 = batch_norm(name='d_k_bn0_0')
        self.d_bn0_1 = batch_norm(name='d_k_bn0_1')
        self.d_bn0_2 = batch_norm(name='d_k_bn0_2')
        self.d_bn1_0 = batch_norm(name='d_k_bn1_0')
        self.d_bn1_1 = batch_norm(name='d_k_bn1_1')
        self.d_bn1_2 = batch_norm(name='d_k_bn1_2')
        self.d_bn1_3 = batch_norm(name='d_k_bn1_3')
        self.d_bn2_0 = batch_norm(name='d_k_bn2_0')
        self.d_bn2_1 = batch_norm(name='d_k_bn2_1')
        self.d_bn2_2 = batch_norm(name='d_k_bn2_2')
        self.d_bn3_0 = batch_norm(name='d_k_bn3_0')
        self.d_bn3_1 = batch_norm(name='d_k_bn3_1')
        self.d_bn3_2 = batch_norm(name='d_k_bn3_2')
        self.d_bn3_3 = batch_norm(name='d_k_bn3_3')
        self.d_bn4_0 = batch_norm(name='d_k_bn4_0')
        self.d_bn4_1 = batch_norm(name='d_k_bn4_1')
        self.d_bn4_2 = batch_norm(name='d_k_bn4_2')
        self.d_bn5   = batch_norm(name='d_k_bn5')
        
        self.dataset_name = config.dataset
        self.checkpoint_dir = config.checkpoint_dir
        self.samples_dir = config.samples_dir
        if not os.path.exists(self.samples_dir+"/"+self.model_dir):
            os.makedirs(self.samples_dir+"/"+self.model_dir)
        if not os.path.exists(self.checkpoint_dir+"/"+self.model_dir):
            os.makedirs(self.checkpoint_dir+"/"+self.model_dir)
        if not os.path.exists(self.checkpoint_dir+"/"+self.model_dir+"/log"):
            os.makedirs(self.checkpoint_dir+"/"+self.model_dir+"/log")
        self.build_model()
    def build_model(self):

        self.subject_num = SUBJECT_NUM_VGG2 
                
        self.input_labels= tf.placeholder(tf.int64, [self.batch_size,], name='positive_labels')
        self.input_filenames = [tf.placeholder(dtype=tf.string) for _ in range(self.batch_size)]
                
        self.sample_images= tf.placeholder(tf.float32, [self.sample_size] + [self.output_size, self.output_size, self.c_dim], name='sample_images')
        self.sample_input_images = tf.placeholder(tf.float32, [1, self.output_size, self.output_size, self.c_dim], name='sample_input_images')
        
        # Networks
        self.images = []          
        for i in range(self.batch_size):
            file_contents = tf.read_file(self.input_filenames[i])
            image = tf.image.decode_jpeg(file_contents, channels=3)
            image = tf.image.resize_images(image, [self.before_crop_size, self.before_crop_size])
            #if self.random_rotate:
            #    image = tf.py_func(random_rotate_image, [image], tf.uint8)
            if (self.padding > 0):
                image = tf.random_crop(image, [self.output_size, self.output_size, 3])



            #if args.random_crop:
            #    image = tf.random_crop(image, [args.image_size, args.image_size, 3])
            #else:
            #    image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
            if self.random_flip:
                image = tf.image.random_flip_left_right(image)
            self.images.append(tf.subtract(tf.div(tf.cast(image, dtype=tf.float32), 127.5) ,1.0))
        self.images = tf.stack(self.images)

        opt = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1)

        self.images_splits = tf.split(self.images, self.num_gpus)
        self.input_labels_splits = tf.split(self.input_labels, self.num_gpus)
        self.d_loss_real_id = []
        self.d_acc = []
        self.d_loss_real_center = []
        tower_grads = []
        for gpu_id in range(self.num_gpus):
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):

                if gpu_id == 0:
                    reuse = False
                else:
                    reuse = True

                self.D_R_id_logits, self.D_R_fx, self.D_R_ln_w = self.discriminator(self.images_splits[gpu_id],
                                                                                    is_reuse=reuse)  # _, self.D_R_logits, _, self.D_R_id_logits, _, self.D_R_pose_logits,_
                self.m_l2 = tf.pow(tf.norm(tf.matmul(self.D_R_fx, self.D_R_ln_w), ord='euclidean', axis=[-2, -1]),
                                   2) * 0.0000001

                self.d_acc_i = slim.metrics.accuracy(tf.argmax(self.D_R_id_logits, 1), self.input_labels_splits[gpu_id],
                                                     weights=100.0)

                #self.D_R_id_logits = AMSoftmax_logit_v2(self.D_R_id_logits, self.D_R_ln_w,
                #                                        label_batch=self.input_labels_splits[gpu_id],
                #                                        nrof_classes=self.subject_num)
                # tf.global_variables_initializer().run()

                # D Loss
                self.d_loss_real_id_i = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.D_R_id_logits,
                                                                   labels=self.input_labels_splits[gpu_id]))

                # self.d_loss_real_center_i, self.real_centers = center_loss(self.D_R_fx, self.input_labels_splits[gpu_id], 0.5, self.subject_num)
                # self.d_loss_real_center_i *= 0.03
                self.d_loss_regularizer = tf.zeros(shape=[])
                self.d_loss_real_center = tf.zeros(shape=[])

                grads = opt.compute_gradients(self.d_loss_real_id_i)
                tower_grads.append(grads)

                self.d_loss_real_id.append(self.d_loss_real_id_i)
                self.d_acc.append(self.d_acc_i)
                # self.d_loss_real_center.append(self.d_loss_real_center_i)
        # self.d_loss_real_center = tf.reduce_mean(self.d_loss_real_center)
        grads = average_gradients(tower_grads)
        self.train_op = opt.apply_gradients(grads)

        self.d_loss_real_id = tf.reduce_mean(self.d_loss_real_id)
        self.d_acc = tf.reduce_mean(self.d_acc)
        self.d_loss = self.d_loss_real_id

        # Sumaries
        tf.summary.scalar("Total loss", self.d_loss)
        tf.summary.scalar("ID - softmax loss", self.d_loss_real_id)
        tf.summary.scalar("Center loss", self.d_loss_real_center)
        tf.summary.scalar("Regularizer loss", self.d_loss_regularizer)
        tf.summary.scalar("Train accuracy", self.d_acc)
        tf.summary.scalar("M - L2 loss", self.m_l2)

        self.summary_op = tf.summary.merge_all()
        #self.summary_writer = tf.summary.FileWriter(self.checkpoint_dir+"/"+self.model_dir+"/log", self.sess.graph)

        self.d_loss = tf.reduce_mean(self.d_loss)
        self.d_acc = tf.reduce_mean(self.d_acc)

        # Vars
        self.t_vars = tf.trainable_variables()
        self.d_vars = [var for var in self.t_vars if not ( 'd_' not in var.name or 'd_k6_id_31239_pai3pi' in var.name or 'd_k6_id_FactoryOne' in var.name or 'd_k6_id_FactoryTwo' in var.name)]
        
        for var in self.d_vars:
            print var.op.name
        self.d_saver = tf.train.Saver(self.d_vars, keep_checkpoint_every_n_hours=.5, max_to_keep=20)
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=.5, max_to_keep = 10)



        '''
        self.d_saver = tf.train.Saver(self.d_vars2, keep_checkpoint_every_n_hours=.5, max_to_keep = 20)

        name_conversion = {
'd_k01_conv/w': 'discriminator/d_k01_conv/weights',
'd_k_bn0_1/beta': 'discriminator/d_k01_conv/BatchNorm/beta',
'd_k_bn0_1/gamma': 'discriminator/d_k01_conv/BatchNorm/gamma',
'd_k02_conv/w': 'discriminator/d_k02_conv/weights',
'd_k_bn0_2/beta': 'discriminator/d_k02_conv/BatchNorm/beta',
'd_k_bn0_2/gamma': 'discriminator/d_k02_conv/BatchNorm/gamma',
'd_k10_conv/w': 'discriminator/d_k03_conv/weights',
'd_k_bn1_0/beta': 'discriminator/d_k03_conv/BatchNorm/beta',
'd_k_bn1_0/gamma': 'discriminator/d_k03_conv/BatchNorm/gamma',
'd_k11_conv/w': 'discriminator/d_k11_conv/weights',
'd_k_bn1_1/beta': 'discriminator/d_k11_conv/BatchNorm/beta',
'd_k_bn1_1/gamma': 'discriminator/d_k11_conv/BatchNorm/gamma',
'd_k12_conv/w': 'discriminator/d_k12_conv/weights',d_vars
'd_k_bn1_2/beta': 'discriminator/d_k12_conv/BatchNorm/beta',
'd_k_bn1_2/gamma': 'discriminator/d_k12_conv/BatchNorm/gamma',
'd_k20_conv/w': 'discriminator/d_k13_conv/weights',
'd_k_bn2_0/beta': 'discriminator/d_k13_conv/BatchNorm/beta',
'd_k_bn2_0/gamma':'discriminator/d_k13_conv/BatchNorm/gamma',
'd_k21_conv/w': 'discriminator/d_k21_conv/weights',
'd_k_bn2_1/beta': 'discriminator/d_k21_conv/BatchNorm/beta',
'd_k_bn2_1/gamma': 'discriminator/d_k21_conv/BatchNorm/gamma',
'd_k22_conv/w': 'discriminator/d_k22_conv/weights',
'd_k_bn2_2/beta': 'discriminator/d_k22_conv/BatchNorm/beta',
'd_k_bn2_2/gamma': 'discriminator/d_k22_conv/BatchNorm/gamma',
'd_k30_conv/w': 'discriminator/d_k23_conv/weights',
'd_k_bn3_0/beta': 'discriminator/d_k23_conv/BatchNorm/beta',
'd_k_bn3_0/gamma': 'discriminator/d_k23_conv/BatchNorm/gamma',
'd_k31_conv/w': 'discriminator/d_k31_conv/weights',
'd_k_bn3_1/beta': 'discriminator/d_k31_conv/BatchNorm/beta',
'd_k_bn3_1/gamma': 'discriminator/d_k31_conv/BatchNorm/gamma',
'd_k32_conv/w': 'discriminator/d_k32_conv/weights',
'd_k_bn3_2/beta': 'discriminator/d_k32_conv/BatchNorm/beta',
'd_k_bn3_2/gamma': 'discriminator/d_k32_conv/BatchNorm/gamma',
'd_k40_conv/w': 'discriminator/d_k33_conv/weights',
'd_k_bn4_0/beta': 'discriminator/d_k33_conv/BatchNorm/beta',
'd_k_bn4_0/gamma': 'discriminator/d_k33_conv/BatchNorm/gamma',
'd_k41_conv/w': 'discriminator/d_k41_conv/weights',
'd_k_bn4_1/beta': 'discriminator/d_k41_conv/BatchNorm/beta',
'd_k_bn4_1/gamma': 'discriminator/d_k41_conv/BatchNorm/gamma',
'd_k42_conv/w': 'discriminator/d_k42_conv/weights',
'd_k_bn4_2/beta': 'discriminator/d_k42_conv/BatchNorm/beta',
'd_k_bn4_2/gamma': 'discriminator/d_k42_conv/BatchNorm/gamma',
'd_k6_id_lin/Matrix': 'discriminator/d_k6_id_lin/weights',
'd_k6_id_lin/bias': 'discriminator/d_k6_id_lin/biases'
}

        def name_in_checkpoint(var):
            return name_conversion[var.op.name]

        variables_to_restore = {name_in_checkpoint(var):var for var in self.d_vars if var.op.name in name_conversion.keys() }
        self.d_saver = tf.train.Saver(variables_to_restore, keep_checkpoint_every_n_hours=1, max_to_keep = 0)
        '''  
                
      
                
    def train(self, config):

        tf.set_random_seed(0)
        np.random.seed(0)
               
        if config.dataset == 'VGG2':
            path_VGG2_l, pid_VGG2_l, path_VGG2_n, pid_VGG2_n = load_VGG2_train_by_list(initial_id_l=0, initial_id_n=0)

            path_all = path_VGG2_n
            pid_all = pid_VGG2_n

            valid_idx = np.random.permutation(len(path_all))

            num_VGG2_n = len(path_all)

            id_dict_n = {}
            for i in range(num_VGG2_n):
                if (pid_all[i] not in id_dict_n):
                    id_dict_n[pid_all[i]] = []
                id_dict_n[pid_all[i]].append(i)

        else:
            data = glob(os.path.join("./data", config.dataset, "*.jpg"))

        gpu_options = tf.GPUOptions(visible_device_list=config.gpu, per_process_gpu_memory_fraction=0.95, allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options))

        # np.random.shuffle(data)
        # d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_loss, var_list=self.t_vars, colocate_gradients_with_ops=True)
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.summary_writer = tf.summary.FileWriter(self.checkpoint_dir + "/" + self.model_dir + "/log",
                                                    self.sess.graph)

        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=.5, max_to_keep=10)
     
        """Train DCGAN"""
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            counter = 0
            print(" [!] Load failed...")

        start_time = time.time()
        
        for epoch in xrange(config.epoch):
            batch_idxs = min(len(valid_idx), config.train_size) // config.batch_size
            for idx in xrange(0, batch_idxs):
                    sub_batch = int(config.batch_size/4)
                    anchor_idx = [0] * sub_batch
                    positive_idx = [0] * sub_batch
                    positive2_idx = [0] * sub_batch
                    positive3_idx = [0] * sub_batch
                    ids_n = id_dict_n.keys()
                    random.shuffle(ids_n)
                    for i in range(sub_batch):
                        idx_list = id_dict_n.get(ids_n[i])
                        anchor_idx[i] = idx_list[random.randint(0, len(idx_list) - 1)]
                        positive_idx[i] = idx_list[random.randint(0, len(idx_list) - 1)]
                        positive2_idx[i] = idx_list[random.randint(0, len(idx_list) - 1)]
                        positive3_idx[i] = idx_list[random.randint(0, len(idx_list) - 1)]

                    batch_positive_labels = [pid_all[batch_file] for batch_file in positive_idx]

                    batch_anchor_images = [path_all[batch_file] for batch_file in anchor_idx]
                    batch_positive_images = [path_all[batch_file] for batch_file in positive_idx]
                    batch_positive2_images = [path_all[batch_file] for batch_file in positive2_idx]
                    batch_positive3_images = [path_all[batch_file] for batch_file in positive3_idx]

                    

                    batch_positive_labels = np.concatenate([batch_positive_labels, batch_positive_labels, batch_positive_labels, batch_positive_labels],axis=0)
                    ffeed_dict={ self.input_labels: batch_positive_labels}
                    for i in range(sub_batch):
                        ffeed_dict[self.input_filenames[i]] = batch_anchor_images[i]
                        ffeed_dict[self.input_filenames[i+sub_batch]] = batch_positive_images[i]
                        ffeed_dict[self.input_filenames[i+sub_batch*2]] = batch_positive2_images[i]
                        ffeed_dict[self.input_filenames[i+sub_batch*3]] = batch_positive3_images[i]


                    '''batch_idx = [p for p in range(0, 240000, 1000000)] #valid_idx[idx*config.batch_size:(idx+1)*config.batch_size]
                    #print("--------------")
                    #print(min(positive_idx))
                    #print(max(positive_idx))

                    #batch_idx = valid_idx[idx*config.batch_size:(idx+1)*config.batch_size]
                    #print(min(valid_idx))
                    #print(max(valid_idx))
                    batch_images = [path_all[batch_file] for batch_file in batch_idx]
                    batch_id_labels = [pid_all[batch_file] for batch_file in batch_idx]

                    ffeed_dict={ self.input_labels: batch_id_labels}
                    for i in range(self.batch_size):
                        ffeed_dict[self.input_filenames[i]] = batch_images[i]'''

                    # Update D network
                    _, summary_str, errD, errD_real_id, errD_real_center, d_acc, errD_regularizer, errD_m_l2 = \
                        self.sess.run([self.train_op, self.summary_op, self.d_loss, self.d_loss_real_id, self.d_loss_real_center, self.d_acc, self.d_loss_regularizer, self.m_l2],feed_dict=ffeed_dict)


                    counter += 1
                    if np.mod(counter, 25) == 1:
                        self.summary_writer.add_summary(summary_str, global_step=counter)
                        print("Epoch: [%2d] [%4d/%4d] time: %4.1f, d_loss: %.4f (id: %.3f (R: %.2f), center: %.3f, regu: %.3f, m_l2: %.3f)" \
                            %(epoch, idx, batch_idxs, time.time() - start_time, errD , errD_real_id, d_acc, errD_real_center, errD_regularizer, errD_m_l2))

                    if np.mod(counter, 1000) == 0:
                        self.save(config.checkpoint_dir, counter)




    def discriminator(self, image, is_reuse=False, is_training = True):

        '''s16 = int(self.output_size/16)
        k0_0 = image
        k0_1 = elu(self.d_bn0_1(conv2d(k0_0, self.df_dim*1, d_h=1, d_w =1, name='d_k01_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k01_prelu')
        k0_2 = elu(self.d_bn0_2(conv2d(k0_1, self.df_dim*2, d_h=1, d_w =1, name='d_k02_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k02_prelu')
        #k0_3 =               maxpool2d(k0_2, k=2, padding='VALID')
        k1_0 = elu(self.d_bn1_0(conv2d(k0_2, self.df_dim*2, d_h=2, d_w =2, name='d_k10_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k10_prelu')
        k1_1 = elu(self.d_bn1_1(conv2d(k1_0, self.df_dim*2, d_h=1, d_w =1, name='d_k11_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k11_prelu')
        k1_2 = elu(self.d_bn1_2(conv2d(k1_1, self.df_dim*4, d_h=1, d_w =1, name='d_k12_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k12_prelu')
        #k1_3 =               maxpool2d(k1_2, k=2, padding='VALID')
        k2_0 = elu(self.d_bn2_0(conv2d(k1_2, self.df_dim*4, d_h=2, d_w =2, name='d_k20_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k20_prelu')
        k2_1 = elu(self.d_bn2_1(conv2d(k2_0, self.df_dim*3, d_h=1, d_w =1, name='d_k21_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k21_prelu')
        k2_2 = elu(self.d_bn2_2(conv2d(k2_1, self.df_dim*6, d_h=1, d_w =1, name='d_k22_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k22_prelu')
        #k2_3 =               maxpool2d(k2_2, k=2, padding='VALID')
        k3_0 = elu(self.d_bn3_0(conv2d(k2_2, self.df_dim*6, d_h=2, d_w =2, name='d_k30_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k30_prelu')
        k3_1 = elu(self.d_bn3_1(conv2d(k3_0, self.df_dim*4, d_h=1, d_w =1, name='d_k31_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k31_prelu')
        k3_2 = elu(self.d_bn3_2(conv2d(k3_1, self.df_dim*8, d_h=1, d_w =1, name='d_k32_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k32_prelu')
        #k3_3 =               maxpool2d(k3_2, k=2, padding='VALID')
        k4_0 = elu(self.d_bn4_0(conv2d(k3_2, self.df_dim*8, d_h=2, d_w =2, name='d_k40_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k40_prelu')
        k4_1 = elu(self.d_bn4_1(conv2d(k4_0, self.df_dim*5, d_h=1, d_w =1, name='d_k41_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k41_prelu')
        k4_2 =     self.d_bn4_2(conv2d(k4_1, self.gfc_dim,  d_h=1, d_w =1, name='d_k42_conv', reuse = is_reuse), train=is_training, reuse = is_reuse)

        k5 = tf.nn.avg_pool(k4_2, ksize = [1, s16, s16, 1], strides = [1,1,1,1],padding = 'VALID')
        k5 = tf.reshape(k5, [-1, self.dfc_dim])
        #if (is_training):
        #    k5 = tf.nn.dropout(k5, keep_prob = 0.6)

        #k6_real = linear(k5, 1,                'd_k6_real_lin')'''

        k5, _ = mobilenet_v2_FR_sz224(image, is_training=True, is_reuse=is_reuse, name='d_MobilenetV2')

        k5_normalized = tf.nn.l2_normalize(k5, 0, 1e-10, name='k5_norm')
        k6_id_VGG2, w_VGG2 = linear_no_bias(k5_normalized, SUBJECT_NUM_VGG2, 'd_k6_id_VGG2', reuse=is_reuse, with_w=True)
        k6_id_DeepCam1, w_deepcam1 = linear_no_bias(k5_normalized, SUBJECT_NUM_DEEPCAM1, 'd_k6_id_DeepCam1', reuse=is_reuse, with_w=True)
        k6_id_DeepCam2, w_deepcam2 = linear_no_bias(k5_normalized, SUBJECT_NUM_DEEPCAM2, 'd_k6_id_DeepCam2', reuse=is_reuse, with_w=True)
        k6_id_DeepCam3, w_deepcam3 = linear_no_bias(k5_normalized, SUBJECT_NUM_DEEPCAM3, 'd_k6_id_DeepCam3', reuse=is_reuse, with_w=True)
        k6_id_FaceWH_JiePai, w_FaceWH_JiePai = linear_no_bias(k5_normalized, SUBJECT_NUM_FACEWH_JIEPAI, 'd_k6_id_FaceWH_JiePai', reuse=is_reuse, with_w=True)
        k6_id_MegaFace, w_MegaFace = linear_no_bias(k5_normalized, SUBJECT_NUM_MEGAFACE, 'd_k6_id_MegaFace', reuse=is_reuse, with_w=True)
        #k6_id_MSCeleb_1M, w_MSCeleb_1M = linear_no_bias(k5_normalized, SUBJECT_NUM_MSCELEB_1M, 'd_k6_id_MSCeleb_1M', reuse=is_reuse, with_w=True)
        k6_id_Wuhan, w_Wuhan = linear_no_bias(k5_normalized, SUBJECT_NUM_WUHAN, 'd_k6_id_Wuhan', reuse=is_reuse, with_w=True)
        k6_id_31239_pai3pi, w_31239_pai3pi = linear_no_bias(k5_normalized, SUBJECT_NUM_31239_PAI3PI, 'd_k6_id_31239_pai3pi', reuse=is_reuse, with_w=True)
        k6_id_FactoryOne, w_FactoryOne = linear_no_bias(k5_normalized, SUBJECT_NUM_FACTORYONE, 'd_k6_id_FactoryOne', reuse=is_reuse, with_w=True)
        k6_id_FactoryTwo, w_FactoryTwo = linear_no_bias(k5_normalized, SUBJECT_NUM_FACTORYTWO, 'd_k6_id_FactoryTwo', reuse=is_reuse, with_w=True)

        k6_id = tf.concat([k6_id_VGG2, k6_id_DeepCam1, k6_id_DeepCam2, k6_id_DeepCam3, k6_id_FaceWH_JiePai, k6_id_MegaFace,
                           k6_id_Wuhan, k6_id_31239_pai3pi, k6_id_FactoryOne, k6_id_FactoryTwo], axis=1)
        k6_w = tf.concat([w_VGG2, w_deepcam1, w_deepcam2, w_deepcam3, w_FaceWH_JiePai, w_MegaFace,
                          w_Wuhan, w_31239_pai3pi, w_FactoryOne, w_FactoryTwo], axis=1)
        #k6_id   = AMSoftmax_logit(k5, id_label, embedding_size=self.dfc_dim, nrof_classes=self.subject_num, name='d_k6_id_lin', reuse=is_reuse)
        #k6_pose = linear(k5, self.pose_dim,    'd_k6_pose_lin')

        return k6_id, k5, k6_w #tf.nn.sigmoid(k6_real), k6_real, tf.nn.softmax(k6_id), k6_id, tf.nn.softmax(k6_pose), k6_pose, k5

     
            
    @property
    def model_dir(self):
        return "%s_%s_%s_%s_%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size, self.gf_dim, self.gfc_dim, self.df_dim, self.dfc_dim)
      
    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

            #self.d_saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            #self.g_saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))


            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")

            return False, 0
