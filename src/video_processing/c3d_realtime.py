import os
import queue
import threading

import numpy as np
import tensorflow as tf

from config.config import get_config
from src.display.single_video import display_video
from src.video_processing.graph_util import ImportGraph
from constant.constants import NUM_FRAMES_PER_CLIP
from src.firebase.push_message import push_message
flags = tf.app.flags
gpu_num = 1
flags.DEFINE_integer('batch_size', 1, 'Batch size.')
flags.DEFINE_integer('num_frame_per_clib', 64, 'Nummber of frames per clib')
flags.DEFINE_integer('crop_size', 224, 'Crop_size')
flags.DEFINE_integer('rgb_channels', 3, 'Channels for input')
flags.DEFINE_integer('classics', 2, 'The num of class')
FLAGS = flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
FRAMES_QUEUE = queue.Queue(maxsize=NUM_FRAMES_PER_CLIP)


def process_images(q, c3d, confidence, name="cam"):
	abnormal_count = 0
	print("processing video ...")
	while True:
		if not q.empty():
			segment = list(q.queue)
			print("segment: ", len(segment))
			if len(segment) >= NUM_FRAMES_PER_CLIP:  # load duoc 64 frames
				rgb_images = np.expand_dims(np.array(segment[-NUM_FRAMES_PER_CLIP:]).astype(np.float32), axis=0)
				predict_score = c3d.run(rgb_images)[0]  # day la cho chay predict
				print("%s score: " % name, predict_score)
				# Nhieu lan phat hien abnormal lien tiep
				if (confidence.confidence > 0.5 or confidence.confidence == 0) and predict_score[1] > 0.5:
					abnormal_count += 1
				else:
					abnormal_count = 0
				confidence.confidence = predict_score[1]#np.argmax(predict_score)
				if abnormal_count >= 6:
					# Tang muc canh bao len 0.2
					confidence.confidence = min(1.0, confidence.confidence + 0.08)
					# Push thong bao toi app
					if abnormal_count % 10 == 0:
						push_message(warning=confidence.confidence)


if __name__ == '__main__':
	# tf.app.run()
	# init = tf.global_variables_initializer()

	c3d_graph = ImportGraph(get_config("model", "c3d"))  # day la cho load frozen model
	capture_thread = threading.Thread(
		target=display_video,
		args=(get_config("video", "camera_1"), FRAMES_QUEUE,))
	process_thread = threading.Thread(
		target=process_images,
		args=(FRAMES_QUEUE, c3d_graph,))

	capture_thread.start()
	process_thread.start()
