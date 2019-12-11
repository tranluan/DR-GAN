"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import csv
import json
import random
import pprint
import scipy.misc
import numpy as np
from glob import glob
import os.path
#import matplotlib.pyplot as plt
from time import gmtime, strftime
from config import DATA_DIR


pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def get_image(image_path, image_size, is_crop=True, is_random_crop = False, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, is_random_crop, resize_w)

def save_images(images, size, image_path, inverse = True):
    if (inverse):
        images = inverse_transform(images)

    return imsave(images, size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    img = merge(images, size)

    #plt.imshow(img)
    #plt.show()
    
    return scipy.misc.imsave(path, img)

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def random_crop(x, crop_h, crop_w=None, with_crop_size=None ):
    if crop_w is None:
        crop_w = crop_h
    if with_crop_size is None:
        with_crop_size = False
    h, w = x.shape[:2]

    j = random.randint(0, h - crop_h)
    i = random.randint(0, w - crop_w)

    if with_crop_size:
        return x[j:j+crop_h, i:i+crop_w,:], j, i
    else:
        return x[j:j+crop_h, i:i+crop_w,:]

def crop(x, crop_h, crop_w, j, i):
    if crop_w is None:
        crop_w = crop_h
    
    return x[j:j+crop_h, i:i+crop_w]


def imresize(img, sz):
    return scipy.misc.imresize(img, sz)

def transform(image, npx=64, is_crop=True, is_random_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        if is_random_crop:
            cropped_image = random_crop(image, npx)
        else:  
            cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.


def to_json(output_path, *layers):
    with open(output_path, "w") as layer_f:
        lines = ""
        for w, b, bn in layers:
            layer_idx = w.name.split('/')[0].split('h')[1]

            B = b.eval()

            if "lin/" in w.name:
                W = w.eval()
                depth = W.shape[1]
            else:
                W = np.rollaxis(w.eval(), 2, 0)
                depth = W.shape[0]

            biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
            if bn != None:
                gamma = bn.gamma.eval()
                beta = bn.beta.eval()

                gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
                beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
            else:
                gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

            if "lin/" in w.name:
                fs = []
                for w in W.T:
                    fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

                lines += """
                    var layer_%s = {
                        "layer_type": "fc", 
                        "sy": 1, "sx": 1, 
                        "out_sx": 1, "out_sy": 1,
                        "stride": 1, "pad": 0,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
            else:
                fs = []
                for w_ in W:
                    fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

                lines += """
                    var layer_%s = {
                        "layer_type": "deconv", 
                        "sy": 5, "sx": 5,
                        "out_sx": %s, "out_sy": %s,
                        "stride": 2, "pad": 1,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
                             W.shape[0], W.shape[3], biases, gamma, beta, fs)
        layer_f.write(" ".join(lines.replace("'","").split()))

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)

def visualize(sess, dcgan, config, option):
  if option == 0:
    z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    save_images(samples, [8, 8], './samples/test_%s.png' % strftime("%Y-%m-%d %H:%M:%S", gmtime()))
  elif option == 1:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      save_images(samples, [8, 8], './samples/test_arange_%s.png' % (idx))
  elif option == 2:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in [random.randint(0, 99) for _ in xrange(100)]:
      print(" [*] %d" % idx)
      z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
      z_sample = np.tile(z, (config.batch_size, 1))
      #z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, './samples/test_gif_%s.gif' % (idx))
  elif option == 3:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, './samples/test_gif_%s.gif' % (idx))
  elif option == 4:
    image_set = []
    values = np.arange(0, 1, 1./config.batch_size)

    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

      image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
      make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

    new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
        for idx in range(64) + range(63, -1, -1)]
    make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)

def load_CFP_test(isFlip = False):
    print('Loading CFP Test...')

    if (isFlip):
        fd = open('data/CFP/CFP_images_96_96_mirror_test.dat')
    else:
        fd = open('data/CFP/CFP_images_96_96_test.dat')
    images = np.fromfile(file=fd,dtype=np.uint8)
    images = images.reshape((-1,96,96)).astype(np.float32)
    images = images/127.5 - 1.
    fd.close()

    fileList = open('data/CFP/CFP_label_test.dat', 'rb');
    reader = csv.reader(fileList);
    pid   = [];

    for row in reader:
        pid.append(row[0])

    fileList.close()

    return images, pid #, pose


def load_CFP_recrop_test():
    print 'Loading CFP ...'
   
    fd = open('data/CFP_recrop/CFP_recrop_images_96_96_test.dat')
    all_images = np.fromfile(file=fd, dtype=np.uint8)
    fd.close()
    all_images = all_images.reshape((-1,96,96,3)).astype(np.float32)
    all_images = all_images/127.5 - 1.
    print '    DONE. Finish loading CFP recrop  test with ' + str(all_images.shape[0]) + ' images'
    return all_images

def load_CFP_frontal_idx():
    print 'Loading CFP ...'
   
    fd = open('data/CFP_recrop/CFP_frontal_idx.dat')
    all_idx = np.fromfile(file=fd, dtype=np.uint32)
    fd.close()
    return all_idx

def load_IJBA_recrop_test(isFlip = False):
    print('Loading IJBA recrop...')

    if (isFlip):
        fd = open('data/IJBA_recrop/IJBA_recrop_images_96_96_test.dat')
    else:
        fd = open('data/IJBA_recrop/IJBA_recrop_images_96_96_test.dat')
    images = np.fromfile(file=fd,dtype=np.uint8)
    fd.close()

    images = images.reshape((-1,96,96,3)).astype(np.float32)
    images = images/127.5 - 1.
    print '    DONE. Finish loading IJBA recrop with ' + str(images.shape[0]) + ' images'

    return images

def load_IJBA_subject_ID():
    print 'Loading IJBA_subject_ID ...'
   
    fd = open('data/IJBA_recrop/IJBA_subject_ID.dat')
    all_id = np.fromfile(file=fd, dtype=np.uint32)
    fd.close()
    print '    DONE. Finish loading IJBA' + str(all_id.shape[0]) + ' ids'
    return all_id

def load_IJBA_training_idx():
    print 'Loading IJBA_training_idx.dat ...'
   
    fd = open('data/IJBA_recrop/IJBA_training_idx.dat')
    all_idx = np.fromfile(file=fd, dtype=np.uint32)
    fd.close()
    print '    DONE. Finish loading IJBA' + str(all_idx.shape[0]) + ' indexes'
    return [idx - 1 for idx in all_idx]


def load_IJBA_feature(fold = 1):
    print('Loading IJBA fold %d features...' % (fold))

    fd = open('data/IJBA_recrop/IJBA_fused_feature_fold%d.dat' % (fold))
    f_features = np.fromfile(file=fd,dtype=np.float32)
    fd.close()
    f_features = f_features.reshape((-1,320))


    return f_features



def load_AFLW_by_list(txtfile = '/media/tranluan/SSD_Crucial/AFLW/AFLW_recrop_fileList.dat', initial_path = '/media/tranluan/SSD_Crucial/AFLW/cropped_pad15'):
    print('Opening AFLW from ' + txtfile)

    fileList = open(txtfile, 'rb');
    reader = csv.reader(fileList);
    filename   = [];
    roll  = [];
    pitch = [];
    yaw   = [];
    for row in reader:
        fname = initial_path + '/'+ row[0]
        if os.path.isfile(fname):
            filename.append(fname)
            roll.append(float(row[1]))
            pitch.append(float(row[2]))
            yaw.append(float(row[3]))
    fileList.close()

    print 'Closed!'
    return filename, roll, pitch, yaw

def load_database_by_list(longtail_txt, normal_txt, initial_path = '', initial_id_l = 0, initial_id_n = 0):
    paths_l = []
    labels_l = []
    print("Opening " + longtail_txt + " ...")
    f_l = open(longtail_txt, "r")
    lines = f_l.readlines()
    if lines != []:
        re_label = 0
        line = lines[0]
        line = line.split(';')
        pr_label = int(line[1])
        for line in lines:
            line = line.split(';')
            paths_l.append(initial_path + line[0])
            cu_label = int(line[1])
            if cu_label != pr_label:
                re_label += 1
            labels_l.append(re_label + initial_id_l)
            pr_label = cu_label
    f_l.close()

    paths_n = []
    labels_n = []
    print("Opening " + normal_txt + " ...")
    f_n = open(normal_txt, "r")
    lines = f_n.readlines()
    if lines != []:
        re_label = 0
        line = lines[0]
        line = line.split(';')
        pr_label = int(line[1])
        for line in lines:
            line = line.split(';')
            paths_n.append(initial_path + line[0])
            cu_label = int(line[1])
            if cu_label != pr_label:
                re_label += 1
            labels_n.append(re_label + initial_id_n)
            pr_label = cu_label
    f_n.close()

    print("Closed!")

    return paths_l, labels_l, paths_n, labels_n


def random_rotate_image(image):
    angle = np.random.uniform(low=-10.0, high=10.0)
    return scipy.misc.imrotate(image, angle, 'bicubic')
