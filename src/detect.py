import argparse
import cv2
import face_recognition
import imutils
import math
import numpy as np
import os
import torch

from deepface import DeepFace
from PIL import Image
from typing import Optional, Tuple
from ultralytics import YOLO

first_slowmo_frame = None
last_slowmo_frame = None

horns_path = "../data/horns.png"
horns = Image.open(horns_path).convert("RGBA")


class HornInfo:
	def __init__(self, box_local_position, resized_image):
		self.box_local_position = box_local_position
		self.resized_image = resized_image


class BoxInfo:
	def __init__(self, x1, y1, width, height, confidence):
		self.x1 = x1
		self.y1 = y1
		self.width = width
		self.height = height
		self.confidence = confidence
		self.person_info = None
		self.horn_infos = []


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
		raise FileNotFoundError(f"մոդելի քաշերը չգտնվեցին: {weights_path}")
	result = YOLO(weights_path).to(device)
	return result


shoplifting_detection_model = load_yolo_model("../data/shoplifting.pt")
person_info_model = load_yolo_model("../data/yolov8n_person_detection.pt")


def create_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', required=True, help="Մուտքային տեսաձայնագրության ֆայլի ճանապարհը")
	parser.add_argument('-o', '--output', required=True, help="Ելքային տեսաձայնագրության ֆայլի ճանապարհը")
	return parser


def load_video(path):
	if not os.path.exists(path):
		raise FileNotFoundError(f"տեսաձայնագրությունը չի գտնվել: {path}")
	result = cv2.VideoCapture(path)
	if not result.isOpened():
		raise IOError("չստացվեց բացել տեսաձայնագրությունը")
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

		rgb_frame = cv2.cvtColor(frame_info.modified_frame, cv2.COLOR_BGR2RGB)
		frame_image = Image.fromarray(rgb_frame)

		for box_info in frame_info.shoplifting_boxes:
			for horn_info in box_info.horn_infos:
				frame_image.paste(horn_info.resized_image,
				                  (box_info.x1 + horn_info.box_local_position[0],
				                   box_info.y1 + horn_info.box_local_position[1]),
				                  horn_info.resized_image)

		frame_info.modified_frame = cv2.cvtColor(np.array(frame_image), cv2.COLOR_RGB2BGR)

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
			            f"Race {max_confidence_person_info[3]}",
			            (60, 120),
			            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
			            (0, 0, 255), 2)

		output_video.write(frame_info.modified_frame)
		if frame_info.is_slowmo:
			for _ in range(2):
				output_video.write(frame_info.modified_frame)

		cv2.imshow("Generating output...", frame_info.modified_frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cv2.destroyWindow("Generating output...")


def detect_person_info(cropped_frame: np.ndarray) -> Optional[Tuple[float, int, str, str]]:
	global max_confidence_person_info

	results = person_info_model(cropped_frame, classes=[0])

	for result in results:
		boxes = result.boxes.cpu().numpy()

		for box in boxes:
			x1, y1, x2, y2 = map(int, box.xyxy[0])
			confidence = float(box.conf[0])

			try:
				analysis = DeepFace.analyze(cropped_frame, actions=['age', 'gender', 'race'], enforce_detection=True)[0]
				person_info = confidence, analysis["age"], analysis["dominant_gender"], analysis["dominant_race"]
				if max_confidence_person_info is None or max_confidence_person_info[0] < confidence:
					max_confidence_person_info = person_info
				return person_info

			except Exception as e:
				print(e)
				return None

	return None


def get_horn_infos(cropped_frame):
	result = []

	rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

	face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

	for landmarks in face_landmarks_list:
		left_eye = landmarks["left_eye"]
		right_eye = landmarks["right_eye"]

		left_x = sum([p[0] for p in left_eye]) / len(left_eye)
		right_x = sum([p[0] for p in right_eye]) / len(right_eye)
		left_y = sum([p[1] for p in left_eye]) / len(left_eye)
		right_y = sum([p[1] for p in right_eye]) / len(right_eye)

		center_x = int((left_x + right_x) / 2)

		face_width = int(abs(right_x - left_x) * 2)

		horns_width = int(face_width * 1.3)
		horns_height = int(horns_width * horns.height / horns.width)

		horns_x = center_x - horns_width // 2
		horns_y = int(min(left_y, right_y)) - horns_height - 10  # + 10
		horns_resized = horns.resize((horns_width, horns_height))

		result.append(HornInfo((horns_x, horns_y), horns_resized))

	return result


def process_shoplifting_box(frame, x, y, w, h, confidence, frame_info):
	box = BoxInfo(x, y, w, h, confidence)
	cropped_frame = frame[y: y + h, x: x + w]
	box.person_info = detect_person_info(cropped_frame)
	box.horn_infos = get_horn_infos(cropped_frame)
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
	global first_slowmo_frame
	global last_slowmo_frame

	result = []

	while True:
		has_frame, frame = video.read()
		if not has_frame:
			break
		frame = imutils.resize(frame, width=800)
		frame_info = FrameInfo(frame)
		frame = process_frame(frame, frame_info)
		frame_info.modified_frame = frame

		if frame_info.shoplifting_boxes:
			frame_index = len(result)
			if first_slowmo_frame is None:
				first_slowmo_frame = frame_index
			last_slowmo_frame = frame_index

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
		raise ValueError("կադրեր չեն գտնվել")

	if first_slowmo_frame is not None:
		slowmo_start = max(0, first_slowmo_frame - int(math.ceil(fps)))
		slowmo_end = min(len(frame_infos), last_slowmo_frame + int(math.ceil(fps)))

		for frame_index in range(slowmo_start, slowmo_end):
			frame_infos[frame_index].is_slowmo = True

	output_video = create_video(args.output, fps,
	                            (frame_infos[0].modified_frame.shape[0], frame_infos[0].modified_frame.shape[1]))
	write_video(output_video, frame_infos)


if __name__ == "__main__":
	try:
		main()
	except Exception as e:
		print(f"սխալ: {str(e)}")
	except:
		print("սխալ: չսպասված սխալ")
