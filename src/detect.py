import argparse
import cv2
import imutils
import numpy as np
import os
import torch

from deepface import DeepFace
from typing import Optional, Tuple
from ultralytics import YOLO


class BoxInfo:
	def __init__(self, x1, y1, width, height, confidence):
		self.x1 = x1
		self.y1 = y1
		self.width = width
		self.height = height
		self.confidence = confidence


class FrameInfo:
	def __init__(self, original_frame):
		self.original_frame = original_frame
		self.modified_frame = self.original_frame
		self.is_slowmo = False
		self.shoplifting_boxes = []


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

max_confidence_person_info = None


def load_yolo_model(weights_path):
	if not os.path.exists(weights_path):
		raise FileNotFoundError(f"model weights not found at: {weights_path}")
	result = YOLO(weights_path).to(device)
	return result


shoplifting_detection_model = load_yolo_model("../data/shoplifting.pt")
person_info_model = load_yolo_model("../data/yolov8n_person_detection.pt")


def create_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', required=True, help="Path to input video file")
	parser.add_argument('-o', '--output', required=True, help="Path to output video file")
	return parser


def load_video(path):
	if not os.path.exists(path):
		raise FileNotFoundError(f"video not found at: {path}")
	result = cv2.VideoCapture(path)
	if not result.isOpened():
		raise IOError("couldn't open webcam or video")
	return result


def create_video(path, fps, frame_shape):
	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	result = cv2.VideoWriter(path, fourcc, fps, (frame_shape[1], frame_shape[0]), True)
	return result


def write_video(output_video, frame_infos):
	for frame_info in frame_infos:
		for box_info in frame_info.shoplifting_boxes:
			cv2.rectangle(frame_info.modified_frame, (box_info.x1, box_info.y1),
			              (box_info.x1 + box_info.width, box_info.y1 + box_info.height), (0, 0, 255), 2)
		center_x = int(box_info.x1 + box_info.width / 2)
		cv2.circle(frame_info.modified_frame, (center_x, box_info.y1), 6, (0, 255, 255), -1)
		confidence_text = f"{np.round(box_info.confidence * 100, 2)}%"
		cv2.putText(frame_info.modified_frame, confidence_text, (box_info.x1 + 10, box_info.y1 - 10),
		            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
		            (0, 0, 255), 2)

		if max_confidence_person_info is not None:
			cv2.putText(frame_info.modified_frame,
			            "Suspect",
			            (30, 30),
			            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
			            (0, 0, 255), 2)
			cv2.putText(frame_info.modified_frame,
			            f"Age {max_confidence_person_info[1]}",
			            (60, 60),
			            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
			            (0, 0, 255), 2)
			cv2.putText(frame_info.modified_frame,
			            f"Gender {max_confidence_person_info[2]}",
			            (60, 90),
			            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
			            (0, 0, 255), 2)
			cv2.putText(frame_info.modified_frame,
			            f"Race = {max_confidence_person_info[3]}",
			            (60, 120),
			            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
			            (0, 0, 255), 2)

		output_video.write(frame_info.modified_frame)
		if frame_info.is_slowmo:
			for _ in range(9):
				output_video.write(frame_info.modified_frame)

		cv2.imshow("Generating output...", frame_info.modified_frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cv2.destroyWindow("Generating output...")


def detect_person_info(croped_frame: np.ndarray) -> Optional[Tuple[float, int, str, str]]:
	global max_confidence_person_info

	results = person_info_model(croped_frame, classes=[0])

	for result in results:
		boxes = result.boxes.cpu().numpy()

		for box in boxes:
			x1, y1, x2, y2 = map(int, box.xyxy[0])
			confidence = float(box.conf[0])

			try:
				analysis = DeepFace.analyze(croped_frame, actions=['age', 'gender', 'race'], enforce_detection=True)[0]
				person_info = confidence, analysis["age"], analysis["dominant_gender"], analysis["dominant_race"]
				if max_confidence_person_info is None or max_confidence_person_info[0] < confidence:
					max_confidence_person_info = person_info

			except Exception as e:
				print(e)
				return None

	return None


def process_shoplifting_box(frame, x, y, w, h, confidence, frame_info):
	box = BoxInfo(x, y, w, h, confidence)
	box.person_info = detect_person_info(frame[y: y + h, x: x + w])
	frame_info.shoplifting_boxes.append(box)


def process_shoplifting_prediction(prediction, frame, frame_info):
	xywh = np.array(prediction.boxes.xywh.cpu()).astype("int32")
	xyxy = np.array(prediction.boxes.xyxy.cpu()).astype("int32")
	cc_data = np.array(prediction.boxes.data.cpu())

	for (x1, y1, _, _), (_, _, w, h), (_, _, _, _, confidence, classification) in zip(xyxy, xywh, cc_data):
		if classification == 1 and confidence > 0.5:
			process_shoplifting_box(frame, x1, y1, w, h, confidence, frame_info)


def process_frame(frame, frame_info):
	shoplifting_prediction = shoplifting_detection_model.predict(frame)[0]
	if shoplifting_prediction.boxes:
		process_shoplifting_prediction(shoplifting_prediction, frame, frame_info)

	return frame


def process_video(video):
	result = []

	while True:
		has_frame, frame = video.read()
		if not has_frame:
			break
		frame = imutils.resize(frame, width=800)
		frame_info = FrameInfo(frame)
		frame = process_frame(frame, frame_info)
		frame_info.modified_frame = frame

		result.append(frame_info)
		cv2.imshow("Processing...", frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cv2.destroyWindow("Processing...")

	return result


def main():
	parser = create_parser()
	args = parser.parse_args()

	video = load_video(args.input)
	fps = video.get(cv2.CAP_PROP_FPS)

	frame_infos = process_video(video)
	if not frame_infos:
		raise ValueError("no frames found")

	output_video = create_video(args.output, fps,
	                            (frame_infos[0].modified_frame.shape[0], frame_infos[0].modified_frame.shape[1]))
	write_video(output_video, frame_infos)


if __name__ == "__main__":
	try:
		main()
	except Exception as e:
		print(f"error: {str(e)}")
	except:
		print("error: unexpected error")
