import argparse
import cv2
import numpy as np
import time
import torch

from imutils.video import VideoStream
from imutils.video import FPS
from torchvision.models import detection

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', type=str, default='',
	help='path to the input image; video stream if not specified')
parser.add_argument('-m', '--model', type=str, default='frcnn-mobilenet',
	choices=['frcnn-mobilenet', 'retinanet'],
	help='name of the object detection model')
parser.add_argument('-l', '--labels', type=str, default='coco.txt',
	help='path to file containing list of categories in COCO dataset')
parser.add_argument('-c', '--confidence', type=float, default=0.7,
	help='minimum probability to filter weak detections')
parser.add_argument('-o', '--offset', type=int, default=15,
	help='offset for class label')
parser.add_argument('-s', '--skip_frames', type=int, default=1,
	help='number of frames in which we run object detector once')
args = parser.parse_args()

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CLASSES = open(args.labels, 'r').read().split('\n')
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
MODELS = {
	'frcnn-mobilenet': detection.fasterrcnn_mobilenet_v3_large_320_fpn,
	'retinanet': detection.retinanet_resnet50_fpn
}

model = MODELS[args.model](
	pretrained=True, 
	progress=True,
	num_classes=len(CLASSES), 
	pretrained_backbone=True).to(device)
model.eval()

def transform(im):
	x = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
	x = x.transpose((2, 0, 1))
	x = np.expand_dims(x, axis=0)
	x = x / 255.0
	x = torch.FloatTensor(x)
	return x

def get_detections(image):
	x = transform(image)
	x = x.to(device)
	return model(x)[0]

def get_class_examples(detections, cls_name):
	def is_probably_in_class(i):
		label = int(detections['labels'][i]-1)
		cls = CLASSES[label]
		confidence = detections['scores'][i]
		return cls == cls_name and confidence > args.confidence
	return [i for i in range(len(detections['boxes'])) if is_probably_in_class(i)]

def plot_dogs(detections, image, cls='person'):
	examples = get_class_examples(detections, cls)
	for i in examples:
		label = int(detections['labels'][i]-1)
		confidence = detections['scores'][i]
		box = detections['boxes'][i].detach().cpu().numpy()
		(startX, startY, endX, endY) = box.astype('int')
		text = '{}: {:.2f}%'.format(cls, confidence * 100)
		print('[INFO] {}'.format(text))
		cv2.rectangle(image, (startX, startY), (endX, endY),
			COLORS[label], 2)
		y = startY - args.offset if startY - args.offset > args.offset else startY + args.offset
		cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[label], 2)
	cv2.imshow('Output', image)

def detect_single_image(path):
	image = cv2.imread(path)
	detections = get_detections(image)
	plot_dogs(detections, image)
	cv2.waitKey(0)

def detect_video_stream():
	n_frames = 0
	
	print('[INFO] starting video stream...')
	vs = VideoStream(src=0).start()
	time.sleep(2.0)
	fps = FPS().start()

	while True:
		frame = vs.read()
		(H, W) = frame.shape[:2]
		if n_frames % args.skip_frames == 0:
			detections = get_detections(frame)
		plot_dogs(detections, frame)
		cv2.imshow('Frame', frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break
		fps.update()

if args.image:
	detect_single_image(args.image)
else:
	detect_video_stream()
