import datetime
import json
import os
import queue

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from matplotlib import pyplot as plt

from constant.constants import NUM_FRAMES_PER_CLIP
from src.video_processing.args import cli
from src.video_processing.network import PifPaf, MonoLoco
from src.video_processing.network.process import preprocess_pifpaf, factory_for_gt_haipham
from src.display.single_video import visualize

# from src.firebase.push_message import push_message
flags = tf.compat.v1.app.flags
gpu_num = 1
flags.DEFINE_integer('batch_size', 1, 'Batch size.')
flags.DEFINE_integer('num_frame_per_clib', 64, 'Nummber of frames per clib')
flags.DEFINE_integer('crop_size', 224, 'Crop_size')
flags.DEFINE_integer('rgb_channels', 3, 'Channels for input')
flags.DEFINE_integer('classics', 2, 'The num of class')
FLAGS = flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
FRAMES_QUEUE = queue.Queue(maxsize=NUM_FRAMES_PER_CLIP)
import urllib3
http = urllib3.PoolManager()
MIN_DISTANCE = 2


def is_crowded(world_loc):
	l = len(world_loc)
	if l < 3:
		return False
	for i in range(0, l-2):
		for j in range(i+1, l-1):
			for k in range(j+1, l):
				if np.linalg.norm(world_loc[i] - world_loc[j]) < MIN_DISTANCE and\
					np.linalg.norm(world_loc[j] - world_loc[k]) < MIN_DISTANCE and \
					np.linalg.norm(world_loc[k] - world_loc[i]) < MIN_DISTANCE:
					return True
	return False


def process_images(q, pifpaf, monoloco, confidence, name="cam", args=None):
	print("processing video ...")
	while True:
		if not q.empty():
			segment = list(q.queue)
			# print("segment: ", len(segment))
			image, processed_image, processed_image_cpu = segment[-1]

			height, width, _ = image.shape
			fields = pifpaf.fields(torch.unsqueeze(processed_image, 0))[0]
			_, _, pifpaf_out = pifpaf.forward(image, processed_image_cpu, fields)

			pil_image = Image.fromarray(image)
			intrinsic_size = [xx * 1.3 for xx in pil_image.size]
			kk, dict_gt = factory_for_gt_haipham(intrinsic_size)  # better intrinsics for mac camera
			# if pifpaf_out:
			boxes, keypoints = preprocess_pifpaf(pifpaf_out, (width, height))
			outputs, varss = monoloco.forward(keypoints, kk)
			dic_out = monoloco.post_process(outputs, varss, boxes, keypoints, kk, dict_gt)
			
			if len(dic_out['xyz_pred']) == 0:
				world_loc = np.array(dic_out['xyz_pred'])
			else:
				world_loc = np.array(dic_out['xyz_pred'])[:, [0, 1, 2]]
			# print ("final, ", world_loc)
			print ("People count %d"%len(world_loc))
			
			image_loc = dic_out['uv_centers']
			confidence.image_loc = image_loc
			
			if False:
				visualize(image, image_loc, 1.0) # cuz image was already resized
				plt.imshow(image)
				plt.show()
			
			package = {
				"counter": {
					"value": len(world_loc),
					"type": "Number"
				},
				"timestamp":{
					"value": datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S"),
					"type": "String"
				},
				"crowd_status":{
					"value": 1 if is_crowded(world_loc) else 0,
					"type": "Number"
				}
			}
			# print(package["crowd_status"])
			x = json.dumps(package)
			# Todo: update db
			http.request(
				'PATCH', "159.65.4.224:1026/v2/entities/badinh2-cam1/attrs",
				headers={'Content-Type': 'application/json'},
				body=x)

		#



if __name__ == '__main__':
	# tf.app.run()
	# init = tf.global_variables_initializer()

	# c3d_graph = ImportGraph(get_config("model", "c3d"))  # day la cho load frozen model
	# args = cli()
	# args.camera = True
	# pifpaf = PifPaf(args)
	# monoloco = MonoLoco(model=args.model, device=args.device)
	#
	# capture_thread = threading.Thread(
	# 	target=display_video,
	# 	args=(get_config("video", "camera_1"), FRAMES_QUEUE, args, 0.4))
	# process_thread = threading.Thread(
	# 	target=process_images,
	# 	args=(FRAMES_QUEUE, pifpaf, monoloco, 0.4))
	#
	# capture_thread.start()
	# process_thread.start()
	from src.display.single_video import data_process
	import cv2
	import queue
	import copy
	args = cli()
	args.camera = True
	args.webcam = True
	args.scale = 0.5
	pifpaf_1 = PifPaf(args)
	args.model = "/mnt/d/Antimatlab/multi-cam/weights/monoloco-190513-1437.pkl"
	monoloco_1 = MonoLoco(model=args.model, device=args.device)
	img = cv2.imread("/mnt/d/Antimatlab/trash-data/trash_cropped_frames/16.30.01_M_3/00727.jpg")
	# img = cv2.imread("/mnt/d/Antimatlab/trash-data/trash_cropped_frames/16.30.01_M_1/00032.jpg")
	q1 = queue.Queue()
	q1.put(data_process(img, args))
	process_images(q1, pifpaf_1, monoloco_1, 0, args=copy.deepcopy(args))
	print(np.linalg.norm(np.array([-5.05583239, -5.77347183, 21.52721024]) - np.array([-3.35129619, -6.30005884, 22.90279388])))
