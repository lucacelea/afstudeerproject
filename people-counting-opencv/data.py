from csv import DictWriter
import os.path
from os import path


field_names = ["EPOCH_TIME", "TOTAL", "DISINFECTED"]


def log_detection(data):
	file_exists = path.exists("detection_log.csv")
	with open('detection_log.csv', 'a',newline='') as detections:
		writer = DictWriter(detections,fieldnames=field_names,delimiter=",")

		if file_exists == False:
			writer.writeheader()

		writer.writerow(data)
		detections.close()
