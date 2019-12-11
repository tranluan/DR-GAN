from __future__ import division
import os
import sys
import time
import csv
import random
from random import randint
from math import floor, radians
from glob import glob
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from six.moves import xrange
from ops import *
from utils import *
from Loss_AMSoftmax import *

SUBJECT_NUM_VGG2 = 8631



class DCGAN(object):
    def __init__(self, sess, config):
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
        self.sess = sess
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

        self.random_flip = True
        self.padding = 0.15
        self.before_crop_size = int(self.output_size * (1 + self.padding))
        self.random_rotate = True

        self.z_dim = config.z_dim
        self.pose_dim = 1
        self.pose_repeat = self.z_dim

        #self.il_dim = il_dim
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
        self.d_bn4_1_l = batch_norm(name='d_k_bn4_1_l')
        self.d_bn4_2_l = batch_norm(name='d_k_bn4_2_l')
        self.d_bn4_1_r = batch_norm(name='d_k_bn4_1_r')
        self.d_bn4_2_r = batch_norm(name='d_k_bn4_2_r')
        self.d_bn4_1_p = batch_norm(name='d_k_bn4_1_pose')
        self.d_bn4_2_p = batch_norm(name='d_k_bn4_2_pose')
        self.d_bn4_1_a = batch_norm(name='d_k_bn4_1_ad')
        self.d_bn4_2_a = batch_norm(name='d_k_bn4_2_ad')                
        self.d_bn5   = batch_norm(name='d_k_bn5')
        
        self.g_bn0_0 = batch_norm(name='g_k_bn0_0')
        self.g_bn0_1 = batch_norm(name='g_k_bn0_1')
        self.g_bn0_2 = batch_norm(name='g_k_bn0_2')
        self.g_bn0_3 = batch_norm(name='g_k_bn0_3')
        self.g_bn1_0 = batch_norm(name='g_k_bn1_0')
        self.g_bn1_1 = batch_norm(name='g_k_bn1_1')
        self.g_bn1_2 = batch_norm(name='g_k_bn1_2')
        self.g_bn1_3 = batch_norm(name='g_k_bn1_3')
        self.g_bn2_0 = batch_norm(name='g_k_bn2_0')
        self.g_bn2_1 = batch_norm(name='g_k_bn2_1')
        self.g_bn2_2 = batch_norm(name='g_k_bn2_2')
        self.g_bn2_3 = batch_norm(name='g_k_bn2_3')
        self.g_bn3_0 = batch_norm(name='g_k_bn3_0')
        self.g_bn3_1 = batch_norm(name='g_k_bn3_1')
        self.g_bn3_2 = batch_norm(name='g_k_bn3_2')
        self.g_bn3_3 = batch_norm(name='g_k_bn3_3')
        self.g_bn4_0 = batch_norm(name='g_k_bn4_0')
        self.g_bn4_1 = batch_norm(name='g_k_bn4_1')
        self.g_bn4_2 = batch_norm(name='g_k_bn4_2')
        self.g_bn4_c = batch_norm(name='g_h_bn4_c')
        self.g_bn5   = batch_norm(name='g_k_bn5')
        
        self.g1_bn0_0 = batch_norm(name='g_h_bn0_0')
        self.g1_bn0_1 = batch_norm(name='g_h_bn0_1')
        self.g1_bn0_2 = batch_norm(name='g_h_bn0_2')        
        self.g1_bn1_0 = batch_norm(name='g_h_bn1_0')
        self.g1_bn1_1 = batch_norm(name='g_h_bn1_1')
        self.g1_bn1_2 = batch_norm(name='g_h_bn1_2')
        self.g1_bn2_0 = batch_norm(name='g_h_bn2_0')
        self.g1_bn2_1 = batch_norm(name='g_h_bn2_1')
        self.g1_bn2_2 = batch_norm(name='g_h_bn2_2')
        self.g1_bn3_0 = batch_norm(name='g_h_bn3_0')
        self.g1_bn3_1 = batch_norm(name='g_h_bn3_1')
        self.g1_bn3_2 = batch_norm(name='g_h_bn3_2')
        self.g1_bn4_0 = batch_norm(name='g_h_bn4_0')
        self.g1_bn4   = batch_norm(name='g_h_bn4')
        self.g1_bn5   = batch_norm(name='g_h_bn5')
        self.dataset_name = config.dataset
        self.checkpoint_dir = config.checkpoint_dir
        self.samples_dir = config.samples_dir
        model_dir = "%s_%s_%s_%s_%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size, self.gf_dim, self.gfc_dim, self.df_dim, self.dfc_dim)
        if not os.path.exists(self.samples_dir+"/"+model_dir):
            os.makedirs(self.samples_dir+"/"+model_dir)
        if not os.path.exists(self.checkpoint_dir+"/"+model_dir):
            os.makedirs(self.checkpoint_dir+"/"+model_dir)
        self.build_model()
    
    def build_model(self):
        
        self.subject_num = SUBJECT_NUM_VGG2 
        self.multi_N = 4

        self.input_labels= tf.placeholder(tf.int64, [self.batch_size,], name='positive_labels')
        self.input_filenames = [tf.placeholder(dtype=tf.string) for _ in range(self.batch_size)]


            
        self.real_positive_labels= tf.placeholder(tf.int64, [self.batch_size,], name='positive_hot_code_labels')
        self.real_negative_labels= tf.placeholder(tf.int64, [self.batch_size,], name='negative_hot_code_labels')
        self.real_anchor_hot_code_poses= tf.placeholder(tf.float32, [self.batch_size, self.pose_dim], name='anchor_hot_code_poses')
        #self.real_positive_hot_code_poses= tf.placeholder(tf.float32, [self.batch_size, self.pose_dim], name='positive_hot_code_poses')
        #self.real_positive2_hot_code_poses= tf.placeholder(tf.float32, [self.batch_size, self.pose_dim], name='positive2_hot_code_poses')
        #self.real_positive3_hot_code_poses= tf.placeholder(tf.float32, [self.batch_size, self.pose_dim], name='positive3_hot_code_poses')
        #self.real_negative_hot_code_poses= tf.placeholder(tf.float32, [self.batch_size, self.pose_dim], name='negative_hot_code_poses')
        self.real_aflw_hot_code_poses= tf.placeholder(tf.float32, [self.batch_size, self.pose_dim], name='anchor_hot_code_poses')

        self.aflw_input_filenames = [tf.placeholder(dtype=tf.string) for _ in range(self.batch_size)]
        self.anchor_input_filenames = [tf.placeholder(dtype=tf.string) for _ in range(self.batch_size)]
        self.positive_input_filenames = [tf.placeholder(dtype=tf.string) for _ in range(self.batch_size)]
        self.positive2_input_filenames = [tf.placeholder(dtype=tf.string) for _ in range(self.batch_size)]
        self.positive3_input_filenames = [tf.placeholder(dtype=tf.string) for _ in range(self.batch_size)]
        self.negative_input_filenames = [tf.placeholder(dtype=tf.string) for _ in range(self.batch_size)]

         
        self.G_anchor_hot_code_poses= tf.placeholder(tf.float32, [self.batch_size, self.pose_dim], name='G_anchor_hot_code_poses')
        self.G_positive_hot_code_poses= tf.placeholder(tf.float32, [self.batch_size, self.pose_dim], name='G_positive_hot_code_poses')
        self.G_positive2_hot_code_poses= tf.placeholder(tf.float32, [self.batch_size, self.pose_dim], name='G_positive2_hot_code_poses')
        self.G_positive3_hot_code_poses= tf.placeholder(tf.float32, [self.batch_size, self.pose_dim], name='G_positive3_hot_code_poses')
        self.G_negative_hot_code_poses= tf.placeholder(tf.float32, [self.batch_size, self.pose_dim], name='G_negative_hot_code_poses')
        self.G_combine_hot_code_poses= tf.placeholder(tf.float32, [self.batch_size, self.pose_dim], name='G_combine_hot_code_poses')
        

        #self.sample_images= tf.placeholder(tf.float32, [self.sample_size] + [self.output_size, self.output_size, self.c_dim], name='sample_images')
        #self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        #self.pose = tf.placeholder(tf.float32, [None, self.pose_dim], name='pose')
        #self.il = tf.placeholder(tf.float32, [None, self.il_dim], name='il')
        #self.sample_input_images = tf.placeholder(tf.float32, [1, self.output_size, self.output_size, self.c_dim], name='sample_input_images')
        #self.k52 = tf.placeholder(tf.float32, [1, self.gfc_dim], name='sample_k52')

        # Networks
        def filename2image(input_filenames, batch_size, isflip = True):
            images = []          
            for i in range(batch_size):
                file_contents = tf.read_file(input_filenames[i])
                image = tf.image.decode_jpeg(file_contents, channels=3)
                image = tf.image.resize_images(image, [self.before_crop_size, self.before_crop_size])
                if self.random_rotate:
                    image = tf.py_func(random_rotate_image, [image], tf.uint8)
                if (self.padding > 0):
                    image = tf.random_crop(image, [self.output_size, self.output_size, 3])
                #if args.random_rotate:
                #image = tf.py_func(facenet.random_rotate_image, [image], tf.uint8)
                #if args.random_crop:
                #    image = tf.random_crop(image, [args.image_size, args.image_size, 3])
                #else:
                #    image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
                if self.random_flip and isflip:
                    image = tf.image.random_flip_left_right(image)
                images.append(tf.subtract(tf.div(tf.cast(image, dtype=tf.float32), 127.5), 1.0))
            return tf.stack(images)

        self.aflw_input_images      = filename2image(self.aflw_input_filenames, self.batch_size, isflip=False)
        self.anchor_input_images    = filename2image(self.anchor_input_filenames, self.batch_size)      
        self.positive_input_images  = filename2image(self.positive_input_filenames, self.batch_size)
        self.positive2_input_images = filename2image(self.positive2_input_filenames, self.batch_size)      
        self.positive3_input_images = filename2image(self.positive3_input_filenames, self.batch_size)      
        self.negative_input_images  = filename2image(self.negative_input_filenames, self.batch_size)      


        
        self.G_encode_anchor, self.G_coefficient_anchor = self.generator_encoder( self.anchor_input_images, is_reuse=False)  
        self.G_decode_anchor = self.generator_decoder( tf.random_uniform( shape=[self.batch_size, self.z_dim], minval=-1, maxval=1), 
                                                       self.G_encode_anchor,  self.G_anchor_hot_code_poses,  is_reuse=False)
        
        self.G_encode_positive, self.G_coefficient_positive= self.generator_encoder( self.positive_input_images, is_reuse=True)
        self.G_encode_positive2, self.G_coefficient_positive2 = self.generator_encoder( self.positive2_input_images, is_reuse=True)
        self.G_encode_positive3, self.G_coefficient_positive3 = self.generator_encoder( self.positive3_input_images, is_reuse=True)
        self.G_encode_negative, _                             = self.generator_encoder( self.negative_input_images, is_reuse=True)

        self.G_encode_combine = tf.div( tf.multiply(self.G_encode_anchor, self.G_coefficient_anchor)       + tf.multiply(self.G_encode_positive, self.G_coefficient_positive) + \
                                        tf.multiply(self.G_encode_positive2, self.G_coefficient_positive2) + tf.multiply(self.G_encode_positive3, self.G_coefficient_positive3) , 
                                        self.G_coefficient_anchor + self.G_coefficient_positive + self.G_coefficient_positive2 + self.G_coefficient_positive3 + 1e-6 )

        self.G_decode_positive = self.generator_decoder( tf.random_uniform( shape=[self.batch_size, self.z_dim], minval=-1, maxval=1), self.G_encode_positive, self.G_positive_hot_code_poses, is_reuse=True)
        self.G_decode_positive2 = self.generator_decoder( tf.random_uniform( shape=[self.batch_size, self.z_dim], minval=-1, maxval=1), self.G_encode_positive2, self.G_positive2_hot_code_poses, is_reuse=True)
        self.G_decode_positive3 = self.generator_decoder( tf.random_uniform( shape=[self.batch_size, self.z_dim], minval=-1, maxval=1), self.G_encode_positive3, self.G_positive3_hot_code_poses, is_reuse=True)
        self.G_decode_negative = self.generator_decoder( tf.random_uniform( shape=[self.batch_size, self.z_dim], minval=-1, maxval=1), self.G_encode_negative, self.G_negative_hot_code_poses, is_reuse=True)
        self.G_decode_combine  = self.generator_decoder( tf.random_uniform( shape=[self.batch_size, self.z_dim], minval=-1, maxval=1), self.G_encode_combine,  self.G_combine_hot_code_poses,  is_reuse=True)

        self.D_all_logits, self.D_all_pose_logits, self.D_all_id_logits, self.D_all_k5_ids, self.D_all_k6_ws = self.discriminator(tf.concat(axis=0,
            values=[self.aflw_input_images,
                    self.anchor_input_images,
                    self.positive_input_images,
                    self.positive2_input_images,
                    self.positive3_input_images,
                    self.negative_input_images,
                    self.G_decode_anchor,
                    self.G_decode_positive,
                    self.G_decode_positive2,
                    self.G_decode_positive3,
                    self.G_decode_combine,
                    self.G_decode_negative]), is_reuse=False)
        
        self.D_R_logits, self.D_G_logits = tf.split(axis=0, num_or_size_splits=2, value=self.D_all_logits)
        self.D_R_id_logits, self.D_G_id_logits = tf.split(axis=0, num_or_size_splits=2, value=self.D_all_id_logits)
        self.D_R_pose_logits, self.D_G_pose_logits = tf.split(axis=0, num_or_size_splits=2, value=self.D_all_pose_logits)

        self.D_R_pose_awlf_logits = tf.slice(self.D_R_pose_logits, [0,0], [self.batch_size,-1])
        self.D_R_id_casia_logits  = tf.slice(self.D_R_id_logits, [self.batch_size,0], [self.batch_size * 4,-1])
        self.d_acc = slim.metrics.accuracy(tf.argmax(self.D_R_id_casia_logits, 1),
                                           tf.concat(axis=0, values=[ self.real_positive_labels,
                                                                      self.real_positive_labels,
                                                                      self.real_positive_labels,
                                                                      self.real_positive_labels]), weights=100.0)
        self.D_R_id_casia_logits = AMSoftmax_logit_v2(
            self.D_R_id_casia_logits,
            self.D_all_k6_ws,
            label_batch=tf.concat(axis=0, values=[self.real_positive_labels,
                    self.real_positive_labels,
                    self.real_positive_labels,
                    self.real_positive_labels]), nrof_classes=self.subject_num)
        self.D_G_id_logits = tf.slice(self.D_G_id_logits, [0, 0], [self.batch_size * 5, -1])
        self.g_acc = slim.metrics.accuracy(tf.argmax(self.D_G_id_logits, 1), tf.concat(axis=0,
                                                                                       values=[
                                                                                           self.real_positive_labels,
                                                                                           self.real_positive_labels,
                                                                                           self.real_positive_labels,
                                                                                           self.real_positive_labels,
                                                                                           self.real_positive_labels]), weights=100.0)
        self.D_G_id_logits = AMSoftmax_logit_v2(
            self.D_G_id_logits,
            self.D_all_k6_ws,
            label_batch=tf.concat(axis=0, values=[self.real_positive_labels,
                                                  self.real_positive_labels,
                                                  self.real_positive_labels,
                                                  self.real_positive_labels,
                                                  self.real_positive_labels]), nrof_classes=self.subject_num)


        #self.samplers, _, _ = self.sampler(tf.random_uniform( shape=[self.batch_size, self.z_dim], minval=-1, maxval=1), self.sample_images, self.pose)                                        
        #self.test_sampler, self.sampler_feature, self.sampler_coefficient = self.sampler(tf.random_uniform( shape=[self.batch_size, self.z_dim], minval=-1, maxval=1), self.sample_input_images, self.pose)

        #self.k5, _ = self.generator_encoder(self.sample_input_images, is_reuse=True, is_training = False)
        #self.h0 = self.generator_decoder(self.z, self.k52, self.pose, is_reuse=True, is_training=False)
        #self.k5, _ = self.generator_encoder(self.h0, is_reuse=True, is_training = False)
        #_,_,_,self.k5 = self.discriminator(self.h0, is_reuse=True, is_training = False)     

        #self.D_R_logits3 = tf.concat(0,[self.D_R_logits, D_R_logits2])
        #self.D_R_logits3 = tf.concat(0,[self.D_R_logits, D_R_logits2])
        

        # D Loss
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_R_logits, labels=0.999*tf.ones_like(self.D_R_logits)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_G_logits, labels=tf.zeros_like(self.D_G_logits)))
        #self.d_loss_real_pose = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.D_R_pose_awlf_logits, labels=self.real_aflw_hot_code_poses)) 
        self.d_loss_real_pose = tf.nn.l2_loss(self.D_R_pose_awlf_logits - self.real_aflw_hot_code_poses) / self.pose_dim / self.batch_size

        
        self.d_loss_real_id = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.D_R_id_casia_logits, labels=tf.concat(axis=0,
            values=[self.real_positive_labels,
                    self.real_positive_labels,
                    self.real_positive_labels,
                    self.real_positive_labels])))
        
        self.d_loss =  self.d_loss_real + self.d_loss_fake + self.d_loss_real_id + self.d_loss_real_pose
        

        # G Loss
        self.g_loss_d_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_G_logits, labels=tf.ones_like(self.D_G_logits)))
        self.g_loss_d_id   = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.D_G_id_logits, labels=tf.concat(axis=0,
            values=[self.real_positive_labels,
                    self.real_positive_labels,
                    self.real_positive_labels,
                    self.real_positive_labels,
                    self.real_positive_labels])))
        self.g_loss_d_id_2 = tf.zeros(1)#tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.concat(0,[self.G_anchor_id_logits, self.G_positive_id_logits, self.G_positive2_id_logits, self.G_positive3_id_logits]), tf.concat(0,[self.real_positive_hot_code_labels, self.real_positive_hot_code_labels, self.real_positive_hot_code_labels, self.real_positive_hot_code_labels])))
        self.g_loss_d_pose = tf.nn.l2_loss(self.D_G_pose_logits - tf.concat(axis=0, values=[self.G_anchor_hot_code_poses, self.G_positive_hot_code_poses, self.G_positive2_hot_code_poses, self.G_positive3_hot_code_poses, self.G_combine_hot_code_poses, self.G_negative_hot_code_poses]))  / self.pose_dim / self.batch_size / 6

        self.g_loss =self.g_loss_d_real + self.g_loss_d_id +  self.g_loss_d_pose #+   + 2*self.g_loss_d_id_2

        self.g_acc2 = tf.zeros(1)

        # Sumaries
   
        d_loss_sum = tf.summary.scalar("D loss", self.d_loss)
        d_loss_id_sum = tf.summary.scalar("D_id", self.d_loss_real_id)
        d_acc_id_sum = tf.summary.scalar("D_id_acc", self.d_acc)
        d_loss_pose_sum = tf.summary.scalar("D_pose", self.d_loss_real_pose)
        d_loss_gan_r_sum = tf.summary.scalar("D_gan_r", self.d_loss_real)
        d_loss_gan_f_sum = tf.summary.scalar("D_gan_f", self.d_loss_fake)


        g_loss_sum = tf.summary.scalar("G loss", self.g_loss)
        g_loss_id_sum = tf.summary.scalar("G_id", self.g_loss_d_id)
        g_loss_pose_sum = tf.summary.scalar("G_pose", self.g_loss_d_pose)
        g_loss_gan_sum = tf.summary.scalar("G_gan", self.g_loss_d_real)
        g_acc_id_sum = tf.summary.scalar("G_id_acc", self.g_acc)


        d_real_sum = tf.summary.histogram("d_R", tf.sigmoid(self.D_R_logits))
        d_fake_sum = tf.summary.histogram("d_G", tf.sigmoid(self.D_G_logits))
        g_out_img_summary = tf.summary.image("Generated images", self.G_decode_anchor, max_outputs=5)
        g_in_img_summary = tf.summary.image("Input", self.anchor_input_images, max_outputs=5)


        self.d_summary_op = tf.summary.merge([d_loss_sum, d_loss_id_sum, d_acc_id_sum, d_loss_pose_sum, d_loss_gan_r_sum, d_loss_gan_f_sum, d_real_sum])
        self.g_summary_op = tf.summary.merge([g_loss_sum, g_loss_id_sum, g_acc_id_sum, g_loss_pose_sum, g_loss_gan_sum, d_fake_sum])
        self.g_img_summary_op = tf.summary.merge([g_in_img_summary, g_out_img_summary])
        self.summary_writer = tf.summary.FileWriter(self.checkpoint_dir+"/"+self.model_dir+"/log", self.sess.graph)

          
        #self.d_loss_real_sum = tf.scalar_summary("d_loss_real", self.d_loss_real)
        #self.d_loss_fake_sum = tf.scalar_summary("d_loss_fake", self.d_loss_fake)
                                                    
        #self.d_loss = self.d_loss_real + self.d_loss_fake
        #self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
        #self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.d_vars_load = [var for var in tf.global_variables() if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.g_en_vars = [var for var in tf.global_variables() if 'g_k' in var.name]
        self.g_de_vars = [var for var in tf.global_variables() if 'g_h' in var.name]

        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1, max_to_keep = 20)
        self.d_saver = tf.train.Saver([var for var in self.d_vars_load if not ('pose' in var.name or 'ad' in var.name)], keep_checkpoint_every_n_hours=1, max_to_keep = 0)
        #self.d_saver = tf.train.Saver([var for var in self.d_vars] , keep_checkpoint_every_n_hours=1, max_to_keep = 0)

        def name_in_checkpoint(var):
            print(var.op.name)
            print('d' + var.op.name[1:])
            return 'd' + var.op.name[1:]

        self.variables_to_restore = {name_in_checkpoint(var):var for var in self.g_en_vars}
        self.g_en_saver = tf.train.Saver(self.variables_to_restore, keep_checkpoint_every_n_hours=1, max_to_keep = 0)
        #self.g_de_saver = tf.train.Saver(self.g_de_vars, keep_checkpoint_every_n_hours=1, max_to_keep = 0)
        #self.g_saver = tf.train.Saver([var for var in self.g_vars if not 'g_h5_lin' in var.name], keep_checkpoint_every_n_hours=1, max_to_keep = 0)


    def test_IJBA(self, config, isReload = False, iteration = 0, isFlip = True):
    
        if isReload:
            could_load, iteration = self.load(self.checkpoint_dir)
            if could_load:       
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")   

        if  config.dataset == 'CASIA' or config.dataset == 'CASIA_MTPIE':            
            # Single image IJBA
            images = load_IJBA_recrop_test();

            N = images.shape[0]
            out_features = np.zeros((N, self.gfc_dim), dtype=np.float32)
            out_coefficients = np.zeros((N, 1), dtype=np.float32)



            target_pose = np.zeros([1, self.pose_dim], dtype=np.float)
            #target_pose[0,self.pose_dim/2] = 1.0
            sample_z = np.zeros(shape=(1 , self.z_dim))

            gallery_features = np.zeros((N, self.gfc_dim), dtype=np.float32)
            gallery_coefficients = np.zeros((N, 1), dtype=np.float32)
            #gallery_rotated_images = np.zeros((2, 96, 96, 3), dtype=np.float32)

     
            idxes = range(0,N)
            
            bar = Bar('Processing', max=len(idxes), suffix='%(percent)d%%')

            co = 0

            idx = 0

            for co in range(0,N):
                idx = idxes[co]
                ##co = co + 1
                if (co > len(idxes)):
                    idx = input("Input");


                batch_img = np.array(images[idx,:,:,:]).astype(np.float32).reshape((1,self.output_size, self.output_size, self.c_dim))
                batch_img = np.array(batch_img).reshape(1,96,96,3)

                gallery_rotated_image, gallery_feature, gallery_coefficient = self.sess.run([ self.test_sampler, self.sampler_feature, self.sampler_coefficient], feed_dict={self.z: sample_z, self.sample_input_images:batch_img, self.pose: target_pose} )

                if isFlip:
                    batch_img = np.flip(batch_img, axis = 2)
                    gallery_feature2, gallery_coefficient2 = self.sess.run([ self.sampler_feature, self.sampler_coefficient], feed_dict={self.z: sample_z, self.sample_input_images:batch_img, self.pose: target_pose} )
                    gallery_feature = (gallery_feature + gallery_feature2)/2.0
                    gallery_coefficient = (gallery_coefficient + gallery_coefficient2)/2.0


                gallery_features[idx,:] = gallery_feature
                gallery_coefficients[idx,:] = gallery_coefficient

                    #gallery_features2[i,:] = gallery_feature2
                    #gallery_rotated_images[i,:,:,:] =  np.array(gallery_rotated_image).reshape(1, self.output_size, self.output_size, self.c_dim)
                bar.next()
                #gallery_rotated_images[1,:,:,:] = batch_img
                #gallery_rotated_images[-1,:,:,:] = batch_img2
                #gallery_rotated_images[2,:,:,:] = np.array(images[frontal_idx[idx-5000]-1,:,:,:]).astype(np.float32).reshape((1,self.output_size, self.output_size, self.c_dim))

                #save_images(gallery_rotated_images, [1, gallery_rotated_images.shape[0]],'IJBA_single/rotated_images_{:d}.png'.format(idx))# .format(idx))
                #gallery_feature.tofile('IJBA_single/feature_{:d}.bin'.format(idx))# .format(idx))

            checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
            if isFlip:
                np.savetxt("%s/IJBA_features_iter_%d_flip.txt" % (checkpoint_dir, iteration), gallery_features)
                np.savetxt("%s/IJBA_coefficient_iter_%d_flip.txt" % (checkpoint_dir, iteration), gallery_coefficients)

                print("\nWrited to %s/IJBA_features_iter_%d_flip.txt" % (checkpoint_dir, iteration))       
            else:
                np.savetxt("%s/IJBA_features_iter_%d.txt" % (checkpoint_dir, iteration), gallery_features)
                np.savetxt("%s/IJBA_coefficient_iter_%d.txt" % (checkpoint_dir, iteration), gallery_coefficients)
                print("\nWrited to %s/IJBA_features_iter_%d.txt" % (checkpoint_dir, iteration))
            
        
    def cos_loop(self,matrix, vector):
        """
        Calculating pairwise cosine distance using a common for loop with manually calculated cosine value.
        """
        neighbors = []
        for row in range(matrix.shape[0]):
            vector_norm = np.linalg.norm(vector)
            row_norm = np.linalg.norm(matrix[row,:])
            cos_val = vector.dot(matrix[row,:]) / (vector_norm * row_norm)
            neighbors.append(cos_val)
        return neighbors
             
                
    def train(self, config):

        #tf.set_random_seed(666)
        #np.random.seed(777)
               
        if config.dataset == 'VGG2':
            path_VGG2_l, pid_VGG2_l, path_VGG2_n, pid_VGG2_n = load_VGG2_train_by_list(initial_id_l=0, initial_id_n=0)

            

            path_all_l = path_VGG2_l 
            path_all = path_VGG2_n 

            pid_all = pid_VGG2_n 

            

            path_AFLW, roll_AFLW, pitch_AFLW, yaw_AFLW = load_AFLW_by_list()
            

            num_VGG2 = len(pid_all)
            num_AFLW = len(yaw_AFLW)

            id_dict = {}
            for i in range(num_VGG2):
                if (pid_all[i] not in id_dict):
                    id_dict[pid_all[i]] = []
                id_dict[pid_all[i]].append(i)

     
            hot_code_pose_AFLW = np.zeros((num_AFLW, self.pose_dim), dtype=np.float)
            for i, label in enumerate(yaw_AFLW):
                #pose_code = int(floor((yaw_AFLW[i]+97.5)/15))
                #if (pose_code < 0):
                #    pose_code = 0
                #elif (pose_code >= self.pose_dim):
                #    pose_code = self.pose_dim-1
                #hot_code_pose_AFLW[i, pose_code] = 1.0
                hot_code_pose_AFLW[i,0] = radians(yaw_AFLW[i])

        else:
            data = glob(os.path.join("./data", config.dataset, "*.jpg"))
        #np.random.shuffle(data)
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        tf.global_variables_initializer().run()
        
        #self.g_sum = tf.merge_summary([self.z_sum, self.d__sum, 
        #    self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        #self.d_sum = tf.merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        #self.writer = tf.train.SummaryWriter("./logs", self.sess.graph)
        sample_z = np.random.uniform(-1, 1, size=(self.sample_size , self.z_dim))
                

        """ Sample images """
        '''
        if config.dataset == 'mnist':
            sample_images = data_X[0:self.sample_size]
            sample_labels = data_y[0:self.sample_size]
        elif config.dataset == 'CASIA' or config.dataset == 'CASIA_MTPIE':
            STEP = 100
            sample_files = range(0, self.sample_size)
            for i in range(0, self.sample_size):
                sample_files[i] = int((i/self.pose_dim)*STEP+i)

            sample = [random_crop(images_CASIA[sample_file,:,:], self.output_size, self.output_size) for sample_file in sample_files]
            sample_images = np.array(sample).astype(np.float32)/127.5-1
      
            sample_pose = np.zeros((self.sample_size, self.pose_dim), dtype=np.float)
            for i in range(self.sample_size):
                sample_pose[i] = radians( i - 80.0)

            ##save_images(sample_images, [8, 8], './{:s}/{:s}/batch_sample.png'.format(self.sample_dir, self.model_dir))
        elif config.dataset == 'MTPIE':
            sample_labels, sample_images, sample_images_wrong = load_MTPIE_samples();
            sample_pose = np.zeros((64, self.pose_dim), dtype=np.float)
            for i in range(1,self.sample_size):
                sample_images[i,:,:,:] = sample_images[i / self.pose_dim,:,:,:]
            for i in range(self.sample_size):
                sample_pose[i,i % self.pose_dim] = 1.0
            sample_il = np.zeros((self.sample_size, self.il_dim), dtype=np.float)
            for i in range(self.sample_size):
                sample_il[i,i % selfil_dim] = 1.0
        else:
            sample_files = data[0:self.sample_size]
            sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop, resize_w=self.output_size, is_grayscale = self.is_grayscale) for sample_file in sample_files]
            if (self.is_grayscale):
                sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
            else:
                sample_images = np.array(sample).astype(np.float32)
        '''

        self.d_saver = tf.train.Saver([var for var in self.d_vars_load if not ('pose' in var.name or 'ad' in var.name)], keep_checkpoint_every_n_hours=1, max_to_keep=0)
        self.g_en_saver = tf.train.Saver(self.variables_to_restore, keep_checkpoint_every_n_hours=1, max_to_keep=0)
        #self.g_de_saver = tf.train.Saver(self.g_de_vars, keep_checkpoint_every_n_hours=1, max_to_keep=0)
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1, max_to_keep=20)

        """Train DCGAN"""
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            counter = 0
            print(" [!] Load failed...")
        #self.d_saver.restore(self.sess, "/media/tranluan/CropFaceData/Repos/DR-GAN_Bangjie_0524/checkpoints/DRGAN_n4/0525/DCGAN.model-789000")
        #self.g_de_saver.restore(self.sess, "/media/tranluan/CropFaceData/Repos/DR-GAN_Bangjie_0524/checkpoints/DRGAN_n4/0525/DCGAN.model-789000")


        start_time = time.time()
        
        for epoch in xrange(config.epoch):
            valid_idx = np.random.permutation(num_VGG2)
            batch_idxs = min(len(valid_idx), config.train_size) // (config.batch_size *  self.multi_N)
            
            for idx in xrange(0, batch_idxs):
                if config.dataset == 'VGG2':
                    ffeed_dict={}

                    #Random pose code
                    batch_random_anchor_hot_code_poses    = np.random.normal(loc=0, scale=radians(45.0), size=[config.batch_size, self.pose_dim]) #np.random.uniform(low=radians(-90.0), high=radians(90.0), size=[config.batch_size, self.pose_dim])
                    batch_random_positive_hot_code_poses  = np.random.uniform(low=radians(-90.0), high=radians(90.0), size=[config.batch_size, self.pose_dim])
                    batch_random_positive2_hot_code_poses = np.random.uniform(low=radians(-90.0), high=radians(90.0), size=[config.batch_size, self.pose_dim])
                    batch_random_positive3_hot_code_poses = np.random.uniform(low=radians(-90.0), high=radians(90.0), size=[config.batch_size, self.pose_dim])
                    batch_random_negative_hot_code_poses  = np.random.uniform(low=radians(-90.0), high=radians(90.0), size=[config.batch_size, self.pose_dim])
                    batch_random_combine_hot_code_poses   = np.random.uniform(low=radians(-90.0), high=radians(90.0), size=[config.batch_size, self.pose_dim])

                    #batch_random_combine_hot_code_poses   = np.random.uniform(low=radians(-90.0), high=radians(90.0), size=[config.batch_size, self.pose_dim])


                    ffeed_dict[self.G_anchor_hot_code_poses]    = batch_random_anchor_hot_code_poses
                    ffeed_dict[self.G_positive_hot_code_poses]  = batch_random_positive_hot_code_poses
                    ffeed_dict[self.G_positive2_hot_code_poses] = batch_random_positive2_hot_code_poses
                    ffeed_dict[self.G_positive3_hot_code_poses] = batch_random_positive3_hot_code_poses
                    ffeed_dict[self.G_negative_hot_code_poses]  = batch_random_negative_hot_code_poses
                    ffeed_dict[self.G_combine_hot_code_poses]   = batch_random_combine_hot_code_poses


                    #AFLW
                    batch = []
                    aflw_idx = np.random.random_integers(0, num_AFLW-1, self.batch_size)
                    batch_aflw_images = [path_AFLW[batch_file]  for batch_file in aflw_idx]
                    batch_aflw_hot_code_poses    = [hot_code_pose_AFLW[batch_file,:] for batch_file in aflw_idx]
                    ffeed_dict[self.real_aflw_hot_code_poses] = batch_aflw_hot_code_poses

                    for i in range(self.batch_size):
                        ffeed_dict[self.aflw_input_filenames[i]] = batch_aflw_images[i]

    
                    #VGG2
                    anchor_idx = [0] * config.batch_size
                    positive_idx = [0] * config.batch_size
                    positive2_idx = [0] * config.batch_size
                    positive3_idx = [0] * config.batch_size
                    negative_idx = [0] * config.batch_size

                    ids = id_dict.keys()
                    random.shuffle(ids)
                    for i in range(config.batch_size):
                        idx_list = id_dict.get(ids[i])
                        anchor_idx[i]   = idx_list[random.randint(0, len(idx_list)-1)]
                        positive_idx[i] = idx_list[random.randint(0, len(idx_list)-1)]
                        positive2_idx[i] = idx_list[random.randint(0, len(idx_list)-1)]
                        positive3_idx[i] = idx_list[random.randint(0, len(idx_list)-1)]


                    for i in range(config.batch_size):
                        negative_idx[i] = random.randint(0, len(path_all_l)-1)




                    batch_positive_labels  = [pid_all[batch_file] for batch_file in positive_idx]
                    batch_negative_labels  = np.zeros(config.batch_size)

                    batch_anchor_images    = [path_all[batch_file]  for batch_file in anchor_idx]
                    batch_positive_images  = [path_all[batch_file]  for batch_file in positive_idx]
                    batch_positive2_images = [path_all[batch_file]  for batch_file in positive2_idx]
                    batch_positive3_images = [path_all[batch_file]  for batch_file in positive3_idx]
                    #batch_positive4_images = [path_all[batch_file]  for batch_file in positive4_idx]
                    batch_negative_images  = [path_all_l[batch_file]  for batch_file in negative_idx]


                    ffeed_dict[self.real_positive_labels]       = batch_positive_labels
                    ffeed_dict[self.real_negative_labels]       = batch_negative_labels

                    for i in range(self.batch_size):
                        ffeed_dict[   self.anchor_input_filenames[i]] = batch_anchor_images[i]
                        ffeed_dict[ self.positive_input_filenames[i]] = batch_positive_images[i]
                        ffeed_dict[self.positive2_input_filenames[i]] = batch_positive2_images[i]
                        ffeed_dict[self.positive3_input_filenames[i]] = batch_positive3_images[i]
                        ffeed_dict[ self.negative_input_filenames[i]] = batch_negative_images[i]
                    

                    #batch_anchor_hot_code_poses    = [hot_code_pose[batch_file,:] for batch_file in anchor_idx]
                    #batch_positive_hot_code_poses  = [hot_code_pose[batch_file,:] for batch_file in positive_idx]
                    #batch_positive2_hot_code_poses  = [hot_code_pose[batch_file,:] for batch_file in positive2_idx]
                    #batch_positive3_hot_code_poses  = [hot_code_pose[batch_file,:] for batch_file in positive3_idx]
                    #batch_negative_hot_code_poses  = [hot_code_pose[batch_file,:] for batch_file in negative_idx]

                   
                    
                try:
                
                    if counter % (2) == 0:
                        # Update D network
                        _, summary_str = self.sess.run([d_optim,  self.d_summary_op], feed_dict= ffeed_dict)
                        self.summary_writer.add_summary(summary_str, global_step=counter)


                        self.sess.run([g_optim], feed_dict= ffeed_dict)

                    else:
                        # Update G network
                        _, summary_str = self.sess.run([g_optim, self.g_summary_op], feed_dict= ffeed_dict)
                        self.summary_writer.add_summary(summary_str, global_step=counter)
                except:
                    #print("Unexpected error:" + sys.exc_info()[0])
                    continue

                counter += 1
                if np.mod(counter, 10) == 1:
                    _, errD_real_pose, errD_real_id, errD_real, errD_fake, errD, errG_d_id, errG_d_id2, errG_d_pose, errG_d_id, errG_d_real, errG, d_acc, g_acc  = \
                    self.sess.run([g_optim, self.d_loss_real_pose, self.d_loss_real_id, self.d_loss_real, self.d_loss_fake, self.d_loss, self.g_loss_d_id, self.g_loss_d_id_2, self.g_loss_d_pose, self.g_loss_d_id, self.g_loss_d_real, self.g_loss, self.d_acc, self.g_acc], feed_dict= ffeed_dict)
 
                    print("Epoch: [%2d] [%4d/%4d] time: %4.1f, d_loss: %.4f (pose:%.3f, id: %.3f (R: %.2f), err_G: %.3f (real: %.2f, fake: %.2f) ), g_loss: %.4f (real:%.3f, id:%.3f (R: %.2f), pose:%.3f)" \
                       %(epoch, idx, batch_idxs, time.time() - start_time, errD, errD_real_pose, errD_real_id, d_acc, errD_real + errD_fake, errD_real, errD_fake, errG, errG_d_real, errG_d_id, g_acc, errG_d_pose ))

                if np.mod(counter, 500) == 1:
                    summary_str = self.sess.run(self.g_img_summary_op, feed_dict= ffeed_dict)
                    self.summary_writer.add_summary(summary_str, global_step=counter)
                
                '''    
                if np.mod(counter, 500) == 1:
                    
                    samples = self.sess.run( self.samplers, feed_dict={self.z: sample_z, self.sample_images: sample_images, self.pose: sample_pose} )
                    if (counter == 1):
                        save_images(sample_images, [13, 13],'./{:s}/{:s}/train_in_{:02d}_{:06d}.png'.format(self.samples_dir, model_dir, epoch, idx))

                    save_images(samples, [13, 13], './{:s}/{:s}/train_{:02d}_{:06d}.png'.format(self.samples_dir, model_dir, epoch, idx))
                    #save_images(batch_images, [8, 8],
                    #            './samples/input_{:02d}_{:04d}.png'.format(epoch, idx))
                    #print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                '''
                if np.mod(counter, 1000) == 0:
                    self.save(config.checkpoint_dir, counter)
                #if np.mod(counter, 5000) == 0 and (config.dataset == 'CASIA_MTPIE'):
                #    #with open('./{:s}/{:s}/test_recogn.txt'.format(self.checkpoint_dir, self.model_dir), "a") as myfile:
                #    self.test_IJBA(config, isReload = False, iteration = counter)

    def discriminator(self, image,  is_reuse=False, is_training = True):

        s16 = int(self.output_size/16)
        images = tf.split(axis=0, num_or_size_splits=2, value=image)
        k4_0 = []
        for ii in range(0, 2):
            if ii == 0:
                is_reuse_l = False
            else:
                is_reuse_l = True
            k0_0 = images[ii]
            k0_1 = elu(self.d_bn0_1(conv2d(k0_0, self.df_dim*1, d_h=1, d_w =1, name='d_k01_conv', reuse = is_reuse_l), train=is_training, reuse = is_reuse_l), name='d_k01_prelu')
            k0_2 = elu(self.d_bn0_2(conv2d(k0_1, self.df_dim*2, d_h=1, d_w =1, name='d_k02_conv', reuse = is_reuse_l), train=is_training, reuse = is_reuse_l), name='d_k02_prelu')
            #k0_3 =               maxpool2d(k0_2, k=2, padding='VALID')
            k1_0 = elu(self.d_bn1_0(conv2d(k0_2, self.df_dim*2, d_h=2, d_w =2, name='d_k10_conv', reuse = is_reuse_l), train=is_training, reuse = is_reuse_l), name='d_k10_prelu')
            k1_1 = elu(self.d_bn1_1(conv2d(k1_0, self.df_dim*2, d_h=1, d_w =1, name='d_k11_conv', reuse = is_reuse_l), train=is_training, reuse = is_reuse_l), name='d_k11_prelu')
            k1_2 = elu(self.d_bn1_2(conv2d(k1_1, self.df_dim*4, d_h=1, d_w =1, name='d_k12_conv', reuse = is_reuse_l), train=is_training, reuse = is_reuse_l), name='d_k12_prelu')
            #k1_3 =               maxpool2d(k1_2, k=2, padding='VALID')
            k2_0 = elu(self.d_bn2_0(conv2d(k1_2, self.df_dim*4, d_h=2, d_w =2, name='d_k20_conv', reuse = is_reuse_l), train=is_training, reuse = is_reuse_l), name='d_k20_prelu')
            k2_1 = elu(self.d_bn2_1(conv2d(k2_0, self.df_dim*3, d_h=1, d_w =1, name='d_k21_conv', reuse = is_reuse_l), train=is_training, reuse = is_reuse_l), name='d_k21_prelu')
            k2_2 = elu(self.d_bn2_2(conv2d(k2_1, self.df_dim*6, d_h=1, d_w =1, name='d_k22_conv', reuse = is_reuse_l), train=is_training, reuse = is_reuse_l), name='d_k22_prelu')
            #k2_3 =               maxpool2d(k2_2, k=2, padding='VALID')
            k3_0 = elu(self.d_bn3_0(conv2d(k2_2, self.df_dim*6, d_h=2, d_w =2, name='d_k30_conv', reuse = is_reuse_l), train=is_training, reuse = is_reuse_l), name='d_k30_prelu')
            k3_1 = elu(self.d_bn3_1(conv2d(k3_0, self.df_dim*4, d_h=1, d_w =1, name='d_k31_conv', reuse = is_reuse_l), train=is_training, reuse = is_reuse_l), name='d_k31_prelu')
            k3_2 = elu(self.d_bn3_2(conv2d(k3_1, self.df_dim*8, d_h=1, d_w =1, name='d_k32_conv', reuse = is_reuse_l), train=is_training, reuse = is_reuse_l), name='d_k32_prelu')
            #k3_3 =               maxpool2d(k3_2, k=2, padding='VALID')
            k4_0_ii = elu(self.d_bn4_0(conv2d(k3_2, self.df_dim*8, d_h=2, d_w =2, name='d_k40_conv', reuse = is_reuse_l), train=is_training, reuse = is_reuse_l), name='d_k40_prelu')
            k4_0.append(k4_0_ii)
        k4_0 = tf.concat(k4_0, 0)

        # Frontal
        k4_0s = tf.split(axis=0, num_or_size_splits=2, value=k4_0)
        k4_2 = []
        for ii in range(0, 2):
            if ii == 0:
                is_reuse_l = False
            else:
                is_reuse_l = True
            k4_1 = elu(self.d_bn4_1(conv2d(k4_0s[ii], self.df_dim*5, d_h=1, d_w =1, name='d_k41_conv', reuse = is_reuse_l), train=is_training, reuse = is_reuse_l), name='d_k41_prelu')
            k4_2_ii =     self.d_bn4_2(conv2d(k4_1, self.gfc_dim,  d_h=1, d_w =1, name='d_k42_conv', reuse = is_reuse_l), train=is_training, reuse = is_reuse_l)
            k4_2.append(k4_2_ii)
        k4_2 = tf.concat(k4_2, 0)

        k5_id = tf.nn.avg_pool(k4_2, ksize = [1, s16, s16, 1], strides = [1,1,1,1],padding = 'VALID')
        k5_id = tf.reshape(k5_id, [-1, self.dfc_dim]) #tf.nn.dropout(tf.reshape(k5_id, [-1, self.dfc_dim]), keep_prob = 0.6)

        k5_normalized = tf.nn.l2_normalize(k5_id, 0, 1e-10, name='k5_norm')
        k6_id_VGG2, w_VGG2 = linear_no_bias(k5_normalized, SUBJECT_NUM_VGG2, 'd_k6_id_VGG2', reuse=is_reuse, with_w=True)
        
        k6_id = k6_id_VGG2
        k6_w = w_VGG2


        # Left profile
        #k4_1 = elu(self.d_bn4_1_l(conv2d(k4_0, self.df_dim*5, d_h=1, d_w =1, name='d_k41_conv_l', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k41_prelu_l')
        #k4_2 =     self.d_bn4_2_l(conv2d(k4_1, self.gfc_dim,  d_h=1, d_w =1, name='d_k42_conv_l', reuse = is_reuse), train=is_training, reuse = is_reuse)

        #k5 = tf.nn.avg_pool(k4_2, ksize = [1, s16, s16, 1], strides = [1,1,1,1],padding = 'VALID')
        #k5 = tf.nn.dropout(tf.reshape(k5, [-1, self.dfc_dim]), keep_prob = 0.6)
        #k6_id_l   = linear(k5, self.subject_num, 'd_k6_id_lin_l')

        
        # Right profile
        #k4_1 = elu(self.d_bn4_1_r(conv2d(k4_0, self.df_dim*5, d_h=1, d_w =1, name='d_k41_conv_r', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k41_prelu_r')
        #k4_2 =     self.d_bn4_2_r(conv2d(k4_1, self.gfc_dim,  d_h=1, d_w =1, name='d_k42_conv_r', reuse = is_reuse), train=is_training, reuse = is_reuse)

        #k5 = tf.nn.avg_pool(k4_2, ksize = [1, s16, s16, 1], strides = [1,1,1,1],padding = 'VALID')
        #k5 = tf.nn.dropout(tf.reshape(k5, [-1, self.dfc_dim]), keep_prob = 0.6)
        #k6_id_r   = linear(k5, self.subject_num, 'd_k6_id_lin_r')

        # Pose
        k4_1 = elu(self.d_bn4_1_p(conv2d(k4_0, self.df_dim*5, d_h=1, d_w =1, name='d_k41_conv_pose', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k41_prelu_pose')
        k4_2 =     self.d_bn4_2_p(conv2d(k4_1, self.dfc_dim,  d_h=1, d_w =1, name='d_k42_conv_pose', reuse = is_reuse), train=is_training, reuse = is_reuse)

        k5 = tf.nn.avg_pool(k4_2, ksize = [1, s16, s16, 1], strides = [1,1,1,1],padding = 'VALID')
        k5 = tf.nn.dropout(tf.reshape(k5, [-1, self.dfc_dim]), keep_prob = 0.6)
        k6_pose = linear(k5, self.pose_dim,    'd_k6_pose_lin', reuse = is_reuse)

        # Adversarial
        k4_1 = elu(self.d_bn4_1_a(conv2d(k4_0, self.df_dim*5, d_h=1, d_w =1, name='d_k41_conv_ad', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k41_prelu_ad')
        k4_2 =     self.d_bn4_2_a(conv2d(k4_1, self.dfc_dim,  d_h=1, d_w =1, name='d_k42_conv_ad', reuse = is_reuse), train=is_training, reuse = is_reuse)

        k5 = tf.nn.avg_pool(k4_2, ksize = [1, s16, s16, 1], strides = [1,1,1,1],padding = 'VALID')
        k5 = tf.nn.dropout(tf.reshape(k5, [-1, self.dfc_dim]), keep_prob = 0.6)
        k6_real = linear(k5, 1, 'd_k6_ad_lin', reuse = is_reuse)
        

        return k6_real, k6_pose, k6_id, k5_id, k6_w#, k6_id_l, k6_id_r,

    def generator_encoder(self, image,  is_reuse=False, is_training = True):   

        s16 = int(self.output_size/16)
        k0_0 = image
        k0_1 = elu(self.g_bn0_1(conv2d(k0_0, self.gf_dim*1, d_h=1, d_w =1, name='g_k01_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='g_k01_prelu')
        k0_2 = elu(self.g_bn0_2(conv2d(k0_1, self.gf_dim*2, d_h=1, d_w =1, name='g_k02_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='g_k02_prelu')
        #k0_3 =               maxpool2d(k0_2, k=2, padding='VALID')
        k1_0 = elu(self.g_bn1_0(conv2d(k0_2, self.gf_dim*2, d_h=2, d_w =2, name='g_k10_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='g_k10_prelu')
        k1_1 = elu(self.g_bn1_1(conv2d(k1_0, self.gf_dim*2, d_h=1, d_w =1, name='g_k11_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='g_k11_prelu')
        k1_2 = elu(self.g_bn1_2(conv2d(k1_1, self.gf_dim*4, d_h=1, d_w =1, name='g_k12_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='g_k12_prelu')
        #k1_3 =               maxpool2d(k1_2, k=2, padding='VALID')
        k2_0 = elu(self.g_bn2_0(conv2d(k1_2, self.gf_dim*4, d_h=2, d_w =2, name='g_k20_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='g_k20_prelu')
        k2_1 = elu(self.g_bn2_1(conv2d(k2_0, self.gf_dim*3, d_h=1, d_w =1, name='g_k21_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='g_k21_prelu')
        k2_2 = elu(self.g_bn2_2(conv2d(k2_1, self.gf_dim*6, d_h=1, d_w =1, name='g_k22_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='g_k22_prelu')
        #k2_3 =               maxpool2d(k2_2, k=2, padding='VALID')
        k3_0 = elu(self.g_bn3_0(conv2d(k2_2, self.gf_dim*6, d_h=2, d_w =2, name='g_k30_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='g_k30_prelu')
        k3_1 = elu(self.g_bn3_1(conv2d(k3_0, self.gf_dim*4, d_h=1, d_w =1, name='g_k31_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='g_k31_prelu')
        k3_2 = elu(self.g_bn3_2(conv2d(k3_1, self.gf_dim*8, d_h=1, d_w =1, name='g_k32_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='g_k32_prelu')
        #k3_3 =               maxpool2d(k3_2, k=2, padding='VALID')
        k4_0 = elu(self.g_bn4_0(conv2d(k3_2, self.gf_dim*8, d_h=2, d_w =2, name='g_k40_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='g_k40_prelu')
        k4_1 = elu(self.g_bn4_1(conv2d(k4_0, self.gf_dim*5, d_h=1, d_w =1, name='g_k41_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='g_k41_prelu')
        k4_2 =     self.g_bn4_2(conv2d(k4_1, self.gfc_dim,  d_h=1, d_w =1, name='g_k42_conv', reuse = is_reuse), train=is_training, reuse = is_reuse)

        k5 = tf.nn.avg_pool(k4_2, ksize = [1, s16, s16, 1], strides = [1,1,1,1],padding = 'VALID')
        k5 = tf.reshape(k5, [-1, self.gfc_dim])
        if (is_training):
            k5 = tf.nn.dropout(k5, keep_prob = 0.6)

        k4_c =     self.g_bn4_c(conv2d(k4_1, 1,             d_h=1, d_w =1, name='g_h4c_conv', reuse = is_reuse), train=is_training, reuse = is_reuse)
        k5_c = tf.nn.avg_pool(k4_c, ksize = [1, s16, s16, 1], strides = [1,1,1,1],padding = 'VALID')
        k5_c = tf.nn.sigmoid(tf.reshape(k5_c, [-1, 1]))

        #k6 = linear(k5, self.subject_num, 'g_h6_real_lin')
    
        return k5, k5_c#, k6

    def generator_decoder(self, z, k5, pose, is_reuse=False, is_training=True):
        
        n_size = k5.get_shape()
           
        n_size = n_size[0]
        if not n_size == None:
            n_size = int(n_size)

        s = self.output_size
        s2, s4, s8, s16, s32 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32)
                    
        # project `z` and reshape
        yb = tf.reshape(k5, [-1, self.gfc_dim])
        #pb = tf.reshape(tf.tile(pose, [1,self.pose_repeat]) , [-1, self.pose_dim*self.pose_repeat])
        pb = elu(linear(pose, self.pose_repeat, scope= 'g_h6_pose_lin', reuse = is_reuse))
        #ib = tf.reshape(pose, [-1, self.il_dim])
        zb = tf.reshape(z, [-1, self.z_dim])
        zbb = tf.concat(axis=1, values=[zb, pb, yb]) #ib
        self.z_ = linear(zbb, self.gfc_dim*s32*s32, scope= 'g_h5_lin', reuse = is_reuse)
        h5 = tf.reshape(tf.nn.dropout(self.z_, keep_prob = 0.6 ), [-1, s32, s32, self.gfc_dim])
        h5 = elu(self.g1_bn5(h5, train=is_training, reuse = is_reuse), name="g_h5_prelu")
        
        h4_1 = deconv2d(h5, [n_size, s16, s16, self.gf_dim*5], name='g_h4', reuse = is_reuse, with_w=False)
        h4_1 = elu(self.g1_bn4(h4_1, train=is_training, reuse = is_reuse), name="g_h4_prelu")
        h4_0 = deconv2d(h4_1, [n_size, s16, s16, self.gf_dim*8], d_h=1, d_w=1, name='g_40', reuse = is_reuse, with_w=False)
        h4_0 = elu(self.g1_bn4_0(h4_0, train=is_training, reuse = is_reuse), name="g_h40_prelu")

        h3_2 = deconv2d(h4_0, [n_size, s8, s8, self.gf_dim*8], d_h=2, d_w=2, name='g_h32', reuse = is_reuse, with_w=False)
        h3_2 = elu(self.g1_bn3_2(h3_2, train=is_training, reuse = is_reuse), name="g_h32_prelu")
        h3_1 = deconv2d(h3_2, [n_size, s8, s8, self.gf_dim*4], d_h=1, d_w=1, name='g_h31', reuse = is_reuse, with_w=False)
        h3_1 = elu(self.g1_bn3_1(h3_1, train=is_training, reuse = is_reuse), name="g_h31_prelu")
        h3_0 = deconv2d(h3_1, [n_size, s8, s8, self.gf_dim*6], d_h=1, d_w=1, name='g_h30', reuse = is_reuse, with_w=False)
        h3_0 = elu(self.g1_bn3_0(h3_0, train=is_training, reuse = is_reuse), name="g_h30_prelu")


        h2_2 = deconv2d(h3_0, [n_size, s4, s4, self.gf_dim*6], d_h=2, d_w=2, name='g_h22', reuse = is_reuse, with_w=False)
        h2_2 = elu(self.g1_bn2_2(h2_2, train=is_training, reuse = is_reuse), name="g_h22_prelu")
        h2_1 = deconv2d(h2_2, [n_size, s4, s4, self.gf_dim*3], d_h=1, d_w=1, name='g_h21', reuse = is_reuse, with_w=False)
        h2_1 = elu(self.g1_bn2_1(h2_1, train=is_training, reuse = is_reuse), name="g_h21_prelu")
        h2_0 = deconv2d(h2_1, [n_size, s4, s4, self.gf_dim*4], d_h=1, d_w=1, name='g_h20', reuse = is_reuse, with_w=False)
        h2_0 = elu(self.g1_bn2_0(h2_0, train=is_training, reuse = is_reuse), name="g_h20_prelu")

        h1_2 = deconv2d(h2_0, [n_size, s2, s2, self.gf_dim*4], d_h=2, d_w=2, name='g_h12', reuse = is_reuse, with_w=False)
        h1_2 = elu(self.g1_bn1_2(h1_2, train=is_training, reuse = is_reuse), name="g_h12_prelu")
        h1_1 = deconv2d(h1_2, [n_size, s2, s2, self.gf_dim*2], d_h=1, d_w=1, name='g_h11', reuse = is_reuse, with_w=False)
        h1_1 = elu(self.g1_bn1_1(h1_1, train=is_training, reuse = is_reuse), name="g_h11_prelu")
        h1_0 = deconv2d(h1_1, [n_size, s2, s2, self.gf_dim*2], d_h=1, d_w=1, name='g_h10', reuse = is_reuse, with_w=False)
        h1_0 = elu(self.g1_bn1_0(h1_0, train=is_training, reuse = is_reuse), name="g_h10_prelu")


        h0_2 = deconv2d(h1_0, [n_size, s, s, self.gf_dim*2], d_h=2, d_w=2, name='g_h02', reuse = is_reuse, with_w=False)
        h0_2 = elu(self.g1_bn0_2(h0_2, train=is_training, reuse = is_reuse), name="g_h02_prelu")
        h0_1 = deconv2d(h0_2, [n_size, s, s, self.gf_dim], d_h=1, d_w=1, name='g_h01', reuse = is_reuse, with_w=False)
        h0_1 = elu(self.g1_bn0_1(h0_1, train=is_training, reuse = is_reuse), name="g_h01_prelu")
           
        h0 = tf.nn.tanh(deconv2d(h0_1, [n_size, s, s, self.c_dim], d_h=1, d_w=1, name='g_h0', reuse = is_reuse, with_w=False))
            
        return h0


            
    def sampler(self, z, y=None, pose=None, il=None):
                        
        k5, k5_c = self.generator_encoder(y, is_reuse=True, is_training = False)
        h0 = self.generator_decoder(z, k5, pose, is_reuse=True, is_training=False)      
            
        return h0, k5, k5_c
  
    '''        
    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s_%s_%s_%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size, self.gf_dim, self.gfc_dim, self.df_dim, self.dfc_dim)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s_%s_%s_%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size, self.gf_dim, self.gfc_dim, self.df_dim, self.dfc_dim)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            #self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            self.d_saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            self.g_en_saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
    '''


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

            self.d_saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            self.g_en_saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            #self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))


            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")

            return False, 0
