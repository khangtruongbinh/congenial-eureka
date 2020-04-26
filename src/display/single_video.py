# Basic model parameters as external flags.
import cv2
import numpy as np

from src.video_processing.network.process import image_transform


def data_process(tmp_data, args, crop_size=224):
	frame = tmp_data
	image = cv2.resize(frame, None, fx=args.scale, fy=args.scale)
	height, width, _ = image.shape
	# print('resized image size: {}'.format(image.shape))
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	processed_image_cpu = image_transform(image.copy())
	processed_image = processed_image_cpu.contiguous().to(args.device, non_blocking=True)
	return (image, processed_image, processed_image_cpu)


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

def visualize(frame, image_loc, scale):
	for center in image_loc:
		# print (center)
		# print (frame.shape)
		cv2.circle(frame, tuple(map(int, np.array(center)/scale)), 10, (255,0,0), cv2.FILLED)
	# height, width, _ = frame.shape
	# cv2.putText(frame, "Num people %d"%len(image_loc), (width-500, 50),
	# 	cv2.FONT_HERSHEY_SIMPLEX, 2, color, thickness=2, lineType=2)

def display_video(cam1, q, args, confidence):
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
		# print (ret, frame.shape)
		if not ret:
			continue
		if q.full():
			q.get()
			q.get()
		try:
			# image = imutils.resize(frame, height=scale_factor, interpolation=cv2.INTER_NEAREST)

			# image = cv2.resize(frame, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
			image = data_process(frame, args)
		except Exception as e:
			print (e)
			continue
		q.put(image)
		# q.put(image) # simon: why duplicate?
		# print(confidence)
		if confidence > 0.5:
			add_predict(frame, 1, confidence)
		cv2.imshow('original', cv2.resize(frame, None, fx=0.25, fy=0.25))

		cv2.waitKey(int(1000 / fps))