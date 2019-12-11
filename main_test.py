from utils import *
import time
import tensorflow as tf;
from tensorflow.python.ops import data_flow_ops
import os.path


def main():
	'''
	gpu_options = tf.GPUOptions(visible_device_list ="1", per_process_gpu_memory_fraction = 0.5, allow_growth = True)
	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options))

	path_VGG2, pid_VGG2 = load_database_by_list('../data/deepcam1/deepcam1.txt', initial_path = '../data/deepcam1/', initial_id = 0)

	N = 2
	img_paths = [tf.placeholder(dtype=tf.string) for i in range(N)]

	images = []
	for i in range(N):
		file_contents = tf.read_file(img_paths[i])
		image = tf.image.decode_jpeg(file_contents, channels=3)
		image = tf.image.resize_images(image, [96, 96])
		images.append(image)
	images = tf.stack(images)

	#for i, path in enumerate(path_VGG2[1:2]):
	start_time = time.time()

	ffeed_dict = {}
	for i in range(N):
		ffeed_dict[img_paths[i]] = path_VGG2[i]

	image_np = sess.run(images, feed_dict=ffeed_dict)
	print(image_np.shape)
	'''

	txtfile = '/media/user1/RawFaceData/AFLW/AFLW_recrop_fileList.dat'
	initial_path = '/media/user1/RawFaceData/AFLW/cropped_pad15/'

	paths = []
	labels = []
	print("Opening " + txtfile + " ...")
	f = open(txtfile, "r")
	lines = f.readlines()
	count = 0
	count2 = 0
	for line in lines:
		count = count + 1
		line = line.split(',')
		#print(line)
		fname = initial_path + line[0]
		if not os.path.isfile(fname):
			count2 += 1
			print(fname)
	    #paths.append(initial_path + line[0])
	    #labels.append(int(line[1]) + initial_id) 
	f.close()
	print("Closed!")
	print(count)
	print(count2)


	return 0


main()