from csv import DictWriter
import os.path
from os import path
from enum import Enum
from configparser import ConfigParser

field_names = ["SESSION_ID","EPOCH_TIME", "EVENT"]

config_object = ConfigParser()
config_object.read("config.ini")
seperate_session_logging_files = config_object.getboolean("LOGGING","seperate_session_logging_files")

def log_detection(data):
	if seperate_session_logging_files:
		file_name = "detection_log_{}.csv".format(data.get('SESSION_ID'))
	else:
		file_name = "detection_log.csv"
	file_exists = path.exists(file_name)

	with open(file_name, 'a',newline='') as detections:
		writer = DictWriter(detections,fieldnames=field_names,delimiter=",")

		if file_exists == False:
			writer.writeheader()

		writer.writerow(data)
		detections.close()
