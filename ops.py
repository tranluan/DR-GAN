import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
          self.epsilon  = epsilon
          self.momentum = momentum
          self.name = name

    def __call__(self, x, train=True, reuse=False ):
        return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      fused=True,
                      reuse = reuse,
                      is_training=train,
                      scope=self.name)

def binary_cross_entropy(preds, targets, name=None):
    """Computes binary cross entropy given `preds`.

    For brevity, let `x = `, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
    """
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) +
                              (1. - targets) * tf.log(1. - preds + eps)))

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(axis=3, values=[x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])

def conv2d(input_, output_dim, 
           k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02,
           name="conv2d", reuse = False):
    with tf.variable_scope(name, reuse = reuse):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def deconv2d(input_, output_shape,
             k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False, reuse = False):
    with tf.variable_scope(name, reuse = reuse):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def maxpool2d(x, k=2, padding='VALID'):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding=padding)

       
def prelu(x, name, reuse = False):
    shape = x.get_shape().as_list()[-1:]

    with tf.variable_scope(name, reuse = reuse):
        alphas = tf.get_variable('alpha', shape, tf.float32,
                            initializer=tf.constant_initializer(value=0.2))

        return tf.nn.relu(x) + tf.multiply(alphas, (x - tf.abs(x))) * 0.5

def relu(x, name='relu'):
    return tf.nn.relu(x, name)  

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def elu(x, name='elu'):
  return tf.nn.elu(x, name)

def linear(input_, output_size, scope="Linear", reuse = False, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear", reuse = reuse):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def linear_no_bias(input_, output_size, scope="Linear", reuse = False, stddev=0.02, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear", reuse = reuse):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
        if with_w:
            return tf.matmul(input_, matrix), matrix
        else:
            return tf.matmul(input_, matrix)

def triplet_loss(anchor_output, positive_output, negative_output, margin = 0.2 ):
    d_pos = tf.reduce_sum(tf.square(anchor_output - positive_output), 1)
    d_neg = tf.reduce_sum(tf.square(anchor_output - negative_output), 1)

    loss = tf.maximum(0., margin + d_pos - d_neg)
    
    return loss

def cosine_loss(anchor_output, positive_output):
    anchor_output_norm = tf.nn.l2_normalize(anchor_output, 1)
    positive_output_norm = tf.nn.l2_normalize(positive_output, 1)
    loss = 1 - tf.reduce_sum(tf.multiply(anchor_output_norm, positive_output_norm), 1)

    return loss

def cosine_triplet_loss(anchor_output, positive_output, negative_output, margin = 0.2 ):
    anchor_output_norm = tf.nn.l2_normalize(anchor_output, 1)
    positive_output_norm = tf.nn.l2_normalize(positive_output, 1)
    negative_output_norm = tf.nn.l2_normalize(negative_output, 1)

    sim_pos = tf.reduce_sum(tf.multiply(anchor_output_norm, positive_output_norm), 1)
    sim_neg = tf.reduce_sum(tf.multiply(anchor_output_norm, negative_output_norm), 1)

    loss = tf.maximum(0., margin - sim_pos + sim_neg)

    return loss

def center_loss(features, label, alfa, nrof_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)
    loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss, centers

def splineInterpolation(x, x1, matrix):

    N  = int(x.get_shape()[0])
    N1 = int(x1.get_shape()[0])

    distance = tf.square(tf.tile(tf.reshape(x, shape = [N,1, 2]), [1, N1, 1]) - tf.tile(tf.reshape(x1, shape = [1,N1, 2]), [N, 1, 1]))
    distance = tf.sqrt(tf.reduce_sum(distance, axis = 2))

    A = distance
    B = tf.concat(axis=1, values=[tf.ones([x.get_shape()[0], 1], tf.float32), x ])

    return tf.matmul(tf.concat(axis=1, values=[A, B]), matrix)

def bilinear2D(Q, x, y):

    x = tf.clip_by_value(x, clip_value_min = 1, clip_value_max = 126) 
    y = tf.clip_by_value(y, clip_value_min = 1, clip_value_max = 126) 

    x1 = tf.floor(x) 
    x2 = x1 + 1

    y1 = tf.floor(y)
    y2 = y1 + 1

    #i = tf.reshape( tf.concat(1, [x2-x, x-x1]), [-1,1,1,2] )

    k = int(Q.get_shape()[2]) # Number of channels
    q11 = tf.reshape( tf.gather_nd( Q, tf.to_int32(tf.concat(axis=1, values=[x1, y1 ])) ), shape = [-1,k,1,1] ) 
    q12 = tf.reshape( tf.gather_nd( Q, tf.to_int32(tf.concat(axis=1, values=[x1, y2 ])) ), shape = [-1,k,1,1] )
    q21 = tf.reshape( tf.gather_nd( Q, tf.to_int32(tf.concat(axis=1, values=[x2, y1 ])) ), shape = [-1,k,1,1] )
    q22 = tf.reshape( tf.gather_nd( Q, tf.to_int32(tf.concat(axis=1, values=[x2, y2 ])) ), shape = [-1,k,1,1] )

    q = tf.concat( axis=2, values=[  tf.concat(axis=3, values=[q11, q12]) ,   tf.concat(axis=3, values=[q21, q22])   ] )



    #print("q11")
    #print(q.get_shape())


    xx =  tf.tile(   tf.reshape( tf.concat(axis=1, values=[x2-x, x-x1]), shape = [-1,1,1,2] ), multiples = [1,k,1,1])
    yy =  tf.tile(   tf.reshape( tf.concat(axis=1, values=[y2-y, y-y1]), shape = [-1,1,2,1] ), multiples = [1,k,1,1])

    Q_new = tf.matmul(tf.matmul(xx, q), yy)

    #print("Q_new")
    #print(Q_new.get_shape())

    return Q_new

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads
