import argparse
import cv2
import dlib
import numpy as np
import time
import torch

from imutils.video import VideoStream
from imutils.video import FPS
from torchvision.models import detection
from tracker import CentroidTracker

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', type=str, default='',
	help='path to the input image; video stream if not specified')
parser.add_argument('-m', '--model', type=str, default='frcnn-mobilenet',
	choices=['frcnn-mobilenet', 'retinanet'],
	help='name of the object detection model')
parser.add_argument('-l', '--labels', type=str, default='coco.txt',
	help='path to file containing list of categories in COCO dataset')
parser.add_argument('-p', '--prob', type=float, default=0.7,
	help='minimum probability to filter weak detections')
parser.add_argument('-o', '--offset', type=int, default=15,
	help='offset for class label')
parser.add_argument('-s', '--skip_frames', type=int, default=1,
	help='number of frames in which we run object detector once')
parser.add_argument('-c', '--cls', type=str, default='dog',
	help='class of objects to detect or track')
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

def get_class_examples(detections):
	def is_probably_in_class(i):
		cls = CLASSES[int(detections['labels'][i]-1)]
		prob = detections['scores'][i]
		return cls == args.cls and prob > args.prob
	return [i for i in range(len(detections['boxes'])) if is_probably_in_class(i)]

def plot_examples(detections, image):
	examples = get_class_examples(detections)
	for i in examples:
		label = int(detections['labels'][i]-1)
		prob = detections['scores'][i]
		box = detections['boxes'][i].detach().cpu().numpy()
		(x1, y1, x2, y2) = box.astype('int')
		text = '{}: {:.2f}%'.format(args.cls, prob * 100)
		# print('[INFO] {}'.format(text))
		cv2.rectangle(image, (x1, y1), (x2, y2),
			COLORS[label], 2)
		y = y1 - args.offset if y1 - args.offset > args.offset else y1 + args.offset
		cv2.putText(image, text, (x1, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[label], 2)
	cv2.imshow('Output', image)

def detect_single_image(path):
	image = cv2.imread(path)
	detections = get_detections(image)
	plot_examples(detections, image)
	cv2.waitKey(0)

def detect_video_stream():
	n_frames = 0
	trackers = []
	ct = CentroidTracker(max_time_disappeared=40)
	
	print('[INFO] starting video stream...')
	vs = VideoStream(src=0).start()
	time.sleep(2.0)
	fps = FPS().start()

	while True:
		rects = []
		frame = vs.read()
		if n_frames % args.skip_frames == 0:
			detections = get_detections(frame)
			examples = get_class_examples(detections)
			plot_examples(detections, frame)
			for i in examples:
				box = detections['boxes'][i].detach().cpu().numpy()
				rect = box.astype('int')
				rects.append(rect)
				(x1, y1, x2, y2) = rect
				tracker = dlib.correlation_tracker()
				tracker.start_track(frame, dlib.rectangle(x1, y1, x2, y2))
				trackers.append(tracker)
				
		else:
			for tracker in trackers:
				tracker.update(frame)
				pos = tracker.get_position()
				rects.append((
					int(pos.left()),
					int(pos.top()), 
					int(pos.right()), 
					int(pos.bottom())
				))
		
		objects = ct.update(rects)
		for id, (x, y) in objects.items():
			text = "ID {}".format(id)
			cv2.putText(frame, text, (x - args.offset, y - args.offset),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
		cv2.imshow('Output', frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break
		fps.update()
		n_frames += 1
	vs.stop()
	cv2.destroyAllWindows()

if args.image:
	detect_single_image(args.image)
else:
	detect_video_stream()
