import argparse
import imutils
import numpy as np
import os
import cv2
import torch
from ultralytics import YOLO


writer = None
slow_down_ranges = []  # Initialize list to track slow-down ranges

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


def draw_shoplifting_box(frame, x, y, w, h, confidence):
	cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
	center_x = int(x + w / 2)
	cv2.circle(frame, (center_x, y), 6, (0, 0, 255), -1)
	conf_text = f"{np.round(confidence * 100, 2)}%"
	cv2.putText(frame, conf_text, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


def draw_shoplifting_prediction(prediction, frame, frame_count):
	global slow_down_ranges
	status = False

	xywh = np.array(prediction.boxes.xywh.cpu()).astype("int32")
	xyxy = np.array(prediction.boxes.xyxy.cpu()).astype("int32")
	cc_data = np.array(prediction.boxes.data.cpu())

	for (x1, y1, _, _), (_, _, w, h), (_, _, _, _, confidence, classification) in zip(xyxy, xywh, cc_data):
		if classification == 1 and confidence > 0.5:
			status = True
			draw_shoplifting_box(frame, x1, y1, w, h, confidence)

	if status:
		start =  max(0, frame_count - 70)  
		end = frame_count + 25
		new_range = (start, end)
		if should_add_range(new_range, slow_down_ranges):
			slow_down_ranges.append(new_range)
			slow_down_ranges = merge_ranges(slow_down_ranges)

def merge_ranges(ranges):
    if not ranges:
        return []
    ranges.sort()
    merged = [ranges[0]]
    for current in ranges[1:]:
        prev_start, prev_end = merged[-1]
        curr_start, curr_end = current
        if curr_start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, curr_end))
        else:
            merged.append(current)
    return merged

def should_add_range(new_range, existing_ranges):
    new_start, new_end = new_range
    for start, end in existing_ranges:
        if not (new_end < start or new_start > end):  # Overlaps
            return False
    return True

def draw_shoplifting_predictions(predictions, frame, frame_count):
    global slow_down_ranges
    for prediction in predictions:
        if prediction.boxes:
            draw_shoplifting_prediction(prediction, frame, frame_count)
    



def process_frame(frame, frame_count):
    shoplifting_predictions = shoplifting_detection_model.predict(frame)
    draw_shoplifting_predictions(shoplifting_predictions, frame, frame_count)
    return frame


def setup_video_writer(frame: np.ndarray, output_path: str) -> None:
	global writer
	if output_path and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(output_path, fourcc, 25,
									(frame.shape[1], frame.shape[0]), True)


def process_video(video, out_path):
	frame_count = 0
	global slow_down_ranges, writer
	slow_down_ranges = []  # Initialize list to track slow-down ranges
	while True:
		has_frame, frame = video.read()
		if not has_frame:
			break
		frame = imutils.resize(frame, width=800)
		frame_count += 1
		frame = process_frame(frame, frame_count)

		setup_video_writer(frame, out_path)
		if writer is not None:
			# Write frames to output video with slow-down effect
			slow_down_active = any(start <= frame_count <= end for start, end in slow_down_ranges)
			if slow_down_active:
				for _ in range(10):  # Write the same frame multiple times for slow-down
					writer.write(frame)
			else:
				writer.write(frame)

		cv2.imshow("Shoplifting detection", frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break


def main():
	parser = create_parser()
	args = parser.parse_args()

	video = load_video(args.input)
	process_video(video, args.output)


if __name__ == "__main__":
	try:
		main()
	except Exception as e:
		print(f"error: {str(e)}")
	except:
		print("error: unexpected error")
