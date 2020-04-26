# Basic model parameters as external flags.
import cv2
import numpy as np
from PIL import Image


def data_process(tmp_data, crop_size=224):
	img = Image.fromarray(tmp_data.astype(np.uint8))
	if img.width > img.height:
		scale = float(crop_size) / float(img.height)
		img = np.array(cv2.resize(np.array(img), (int(img.width * scale + 1), crop_size))).astype(np.float32)
	else:
		scale = float(crop_size) / float(img.width)
		img = np.array(cv2.resize(np.array(img), (crop_size, int(img.height * scale + 1)))).astype(np.float32)
	crop_x = int((img.shape[0] - crop_size) / 2)
	crop_y = int((img.shape[1] - crop_size) / 2)
	img = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :]
	return img


def add_predict(frame, predict, alert_level, confidence_score=0):
	action_mapping = {0: 'Normal', 1: 'Abnormal'}
	color = (0, 255, 255)
	if predict == 1:
		if confidence_score > alert_level[2]:
			color = (0, 0, 255)
		elif confidence_score > alert_level[1]:
			color = (0, 165, 255)
		elif confidence_score > alert_level[0]:
			color = (0, 255, 255)
	height, width, _ = frame.shape
	cv2.rectangle(frame, (0, 0), (width, height), color, 30)
	cv2.putText(frame, "%s: %.2f" % (action_mapping[predict], confidence_score), (width-500, 50),
		cv2.FONT_HERSHEY_SIMPLEX, 2, color, thickness=2, lineType=2)


def display_video(cam1, q, confidence):
	print("display_video waiting model initialized...")
	cap = cv2.VideoCapture(cam1)
	_, frame = cap.read()
	ret = True
	global fps
	fps = cap.get(cv2.CAP_PROP_FPS)
	# global confidence
	# confidence = 0
	print("display_video ...")
	while ret:
		ret, frame = cap.read()
		if not ret:
			continue
		if q.full():
			q.get()
			q.get()
		try:
			# image = imutils.resize(frame, height=scale_factor, interpolation=cv2.INTER_NEAREST)

			# image = cv2.resize(frame, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
			image = data_process(frame)
		except Exception as e:
			continue
		q.put(image)
		q.put(image)
		print(confidence)
		if confidence > 0.5:
			add_predict(frame, 1, confidence)
		cv2.imshow('original', frame)

		cv2.waitKey(int(1000 / fps))