from typing import Optional, Tuple
from deepface import DeepFace
from ultralytics import YOLO
import numpy as np

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
					
				return (conf_percent, age, gender, race)
			
			except Exception as e:
				return None
		
	return None
