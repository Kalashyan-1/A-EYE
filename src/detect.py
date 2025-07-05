import argparse
import imutils
import numpy as np
import os
import cv2
import torch

from ultralytics import YOLO
from typing import Optional, Tuple
from deepface import DeepFace


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_yolov8_model(weights_path):
	if not os.path.exists(weights_path):
		raise FileNotFoundError(f"model weights not found at: {weights_path}")
	result = YOLO(weights_path).to(device)
	return result

shoplifting_detection_model = load_yolov8_model("../data/shoplifting.pt")
person_info_model = load_yolov8_model("../data/yolov8n_person_detection.pt")


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

def detect_person_info(croped_frame:np.ndarray) -> Optional[Tuple[float, int, str, str]]:
	results = person_info_model(croped_frame, classes=[0])

	for result in results:
		boxes = result.boxes.cpu().numpy()

		for box in boxes:
			x1, y1, x2, y2 = map(int, box.xyxy[0])
			conf = float(box.conf[0])

			try:
				analysis = DeepFace.analyze(croped_frame, actions=['age', 'gender', 'race'], enforce_detection=True)[0]
				#analysis = DeepFace.analyze(
				#    img_path=croped_frame,
				#    actions=['age', 'gender', 'race'],
				#    models={
				#        'age': age_model,
				#        'gender': gender_model,
				#        'race': race_model
				#    },
				#    enforce_detection=True
				#)[0]
				#conf_percent = round(conf * 100, 1)
				#age = analysis["age"]
				#gender = analysis["dominant_gender"]
				#race = analysis["dominant_race"]

				return (conf, analysis["age"], analysis["dominant_gender"], analysis["dominant_race"])

			except Exception as e:
				print(e)
				return None

	return None

def draw_shoplifting_box(frame, x, y, w, h, confidence, person_infos):
	cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
	center_x = int(x + w / 2)
	cv2.circle(frame, (center_x, y), 6, (0, 0, 255), -1)
	conf_text = f"{np.round(confidence * 100, 2)}%"
	cv2.putText(frame, conf_text, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
	person_info = detect_person_info(frame[y : y + h, x : x + w])
	if person_info:
		person_infos.append(person_info)


def draw_shoplifting_prediction(prediction, frame, person_infos):
	xywh = np.array(prediction.boxes.xywh.cpu()).astype("int32")
	xyxy = np.array(prediction.boxes.xyxy.cpu()).astype("int32")
	cc_data = np.array(prediction.boxes.data.cpu())

	for (x1, y1, _, _), (_, _, w, h), (_, _, _, _, confidence, classification) in zip(xyxy, xywh, cc_data):
		if classification == 1 and confidence > 0.5:
			draw_shoplifting_box(frame, x1, y1, w, h, confidence, person_infos)


def draw_shoplifting_predictions(predictions, frame, person_infos):
	for prediction in predictions:
		if prediction.boxes:
			draw_shoplifting_prediction(prediction, frame, person_infos)


def process_frame(frame, person_infos):
	shoplifting_predictions = shoplifting_detection_model.predict(frame)
	draw_shoplifting_predictions(shoplifting_predictions, frame, person_infos)

	return frame


def process_video(video):
	person_infos = []
	while True:
		has_frame, frame = video.read()
		if not has_frame:
			break
		frame = imutils.resize(frame, width=800)

		frame = process_frame(frame, person_infos)

		cv2.imshow("Shoplifting detection", frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	print(max(person_infos, key=lambda x: x[0]))


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




