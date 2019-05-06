#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation, BatchGenerator
from utils import draw_boxes
from frontend import YOLO
import json

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
	description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
	'-c',
	'--conf',
	help='path to configuration file')

argparser.add_argument(
	'-w',
	'--weights',
	help='path to pretrained weights')

argparser.add_argument(
	'-t',
	'--test',
	help='path to test images')

argparser.add_argument(
	'-l',
	'--labels',
	help='path to test labels')

argparser.add_argument(
	'-b',
	'--boxes',
	help='path to test boxes')

def _main_(args):
	config_path  = args.conf
	weights_path = args.weights
	test_dir   = args.test
	label_dir = args.labels
	box_dir = args.boxes

	with open(config_path) as config_buffer:    
		config = json.load(config_buffer)

	###############################
	#   Make the model 
	###############################

	yolo = YOLO(backend             = config['model']['backend'],
				input_size          = config['model']['input_size'], 
				labels              = config['model']['labels'], 
				max_box_per_image   = config['model']['max_box_per_image'],
				anchors             = config['model']['anchors'])

	###############################
	#   Load trained weights
	###############################    

	yolo.load_weights(weights_path)

	###############################
	#   Predict bounding boxes 
	###############################

	test_imgs, test_labels = parse_annotation(label_dir, test_dir, config['model']['labels'])

	generator_config = {
			'IMAGE_H'         : yolo.input_size, 
			'IMAGE_W'         : yolo.input_size,
			'GRID_H'          : yolo.grid_h,  
			'GRID_W'          : yolo.grid_w,
			'BOX'             : yolo.nb_box,
			'LABELS'          : yolo.labels,
			'CLASS'           : len(yolo.labels),
			'ANCHORS'         : yolo.anchors,
			'BATCH_SIZE'      : config['train']['batch_size'],
			'TRUE_BOX_BUFFER' : yolo.max_box_per_image,
        }    

	test_generator = BatchGenerator(test_imgs, 
			generator_config, 
			norm=yolo.feature_extractor.normalize,
			jitter=False)

    ############################################
	# Compute mAP on the validation set
	############################################
	average_precisions = yolo.evaluate(test_generator)     

	# print evaluation
	for label, average_precision in average_precisions.items():
		print(yolo.labels[label], '{:.4f}'.format(average_precision))
	print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))    

	# if test_dir[-4:] == '.mp4':
	# 	video_out = image_path[:-4] + '_detected' + image_path[-4:]
	# 	video_reader = cv2.VideoCapture(image_path)

	# 	nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
	# 	frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
	# 	frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

	# 	video_writer = cv2.VideoWriter(video_out,
	# 							cv2.VideoWriter_fourcc(*'MPEG'), 
	# 							50.0, 
	# 							(frame_w, frame_h))

	# 	for i in tqdm(range(nb_frames)):
	# 		_, image = video_reader.read()

	# 		boxes = yolo.predict(image)
	# 		image = draw_boxes(image, boxes, config['model']['labels'])

	# 		video_writer.write(np.uint8(image))

	# 	video_reader.release()
	# 	video_writer.release()  
	# else:
	files = []
	for (dirpath, dirnames, filenames) in os.walk(test_dir):
		files.extend(filenames)
		break
	if not files:
		print('no images found in directory')
		exit()

	for f in files:
		image_path = '%s/%s' % (test_dir, f)
		image = cv2.imread(image_path)
		boxes = yolo.predict(image)
		image = draw_boxes(image, boxes, config['model']['labels'])

		#print(len(boxes), 'boxes are found')

		cv2.imwrite(box_dir + '/' + f[:-4] + '_detected' + f[-4:], image)

if __name__ == '__main__':
	args = argparser.parse_args()
	_main_(args)
