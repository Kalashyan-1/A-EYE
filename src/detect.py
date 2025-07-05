import argparse
import imutils
import numpy as np
import os
import cv2
import torch
from ultralytics import YOLO


def load_model(weights_path):
	if not os.path.exists(weights_path):
		raise FileNotFoundError(f"model weights not found at: {weights_path}")
	result = YOLO(weights_path).to(device)
	if not hasattr(result, 'predict'):
		raise AttributeError("the shoplifting model has no method 'predict'")
	return result


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
shoplifting_detection_model = load_model("../data/shoplifting.pt")


def create_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', required=True)  # TODO: make positional
	return parser


def load_video(path):
	if not os.path.exists(path):
		raise FileNotFoundError(f"video not found at: {path}")
	result = cv2.VideoCapture(path)
	if not result.isOpened():
		raise IOError("couldn't open webcam or video")
	return result


def draw_shoplifting_box(frame, x, y, w, h, confidence):
	cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
	center_x = int(x + w / 2)
	cv2.circle(frame, (center_x, y), 6, (0, 0, 255), -1)
	conf_text = f"{np.round(confidence * 100, 2)}%"
	cv2.putText(frame, conf_text, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


def draw_shoplifting_prediction(prediction, frame):
	xywh = np.array(prediction.boxes.xywh.cpu()).astype("int32")
	xyxy = np.array(prediction.boxes.xyxy.cpu()).astype("int32")
	cc_data = np.array(prediction.boxes.data.cpu())

	for (x1, y1, _, _), (_, _, w, h), (_, _, _, _, confidence, classification) in zip(xyxy, xywh, cc_data):
		if classification == 1 and confidence > 0.5:
			draw_shoplifting_box(frame, x1, y1, w, h, confidence)


def draw_shoplifting_predictions(predictions, frame):
	for prediction in predictions:
		if prediction.boxes:
			draw_shoplifting_prediction(prediction, frame)


def process_frame(frame):
	shoplifting_predictions = shoplifting_detection_model.predict(frame)
	draw_shoplifting_predictions(shoplifting_predictions, frame)

	return frame


def process_video(video):
	while True:
		has_frame, frame = video.read()
		if not has_frame:
			break
		frame = imutils.resize(frame, width=800)

		frame = process_frame(frame)

		cv2.imshow("Shoplifting detection", frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break


def main():
	parser = create_parser()
	args = parser.parse_args()

	video = load_video(args.input)
	process_video(video)


if __name__ == "__main__":
	try:
		main()
	except Exception as e:
		print(f"error: {str(e)}")
	except:
		print("error: unexpected error")
