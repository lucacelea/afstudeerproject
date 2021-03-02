from configparser import ConfigParser
import os
import cv2
import tkinter as tk
from tkinter import filedialog
from os import path
import subprocess

master = tk.Tk()
master.title("Disinfection Detection")
config_object = ConfigParser()
config_object.read("config.ini")

if path.exists("config.ini"):
    webcam = tk.BooleanVar()
    webcam.set(config_object.getboolean("VIDEOSTREAM","webcam"))
    video_input_file = config_object.get("VIDEOSTREAM","video_input")
    skipped_frames = config_object.get("MODEL","skipped_frames")
    seperate_session_logging_files = tk.BooleanVar()
    seperate_session_logging_files.set(config_object.getboolean("LOGGING","seperate_session_logging_files"))
    model = tk.BooleanVar()
    model.set(config_object.get("MODEL","alt_model_faster_rcnn"))
    detection_confidence = config_object.get("DETECTION","detection_confidence")
    zone_detection_time = config_object.get("DETECTION","zone_detection_time")

else:
    webcam = tk.BooleanVar()
    webcam.set(True)
    video_input_file = ""
    skipped_frames = 2
    seperate_session_logging_files = tk.BooleanVar()
    seperate_session_logging_files.set(True)
    model = tk.BooleanVar()
    model.set(False)
    detection_confidence = 0.4
    zone_detection_time = 2.5



def browseFiles():
    filename = filedialog.askopenfilename(initialdir = "/",
                                          title = "Select a File",
                                          filetypes=[
                    ("Video", ".mp4"),
                    ("Video", ".flv"),
                    ("Video", ".avi"),
                    ("Video", ".avi"),
                ])
    file_entry.delete(0,'end')
    file_entry.insert(0,filename)

file_entry = tk.Entry(master)
file_entry.insert(0,video_input_file)

button_explore = tk.Button(master, 
                        text = "Browse Files",
                        command = browseFiles, padx=10) 


tk.Label(master,text="Videostream",font='Helvetica 12 bold').grid(row=0,padx=(10,10),pady=10)
tk.Label(master, 
         text="Video").grid(row=1,padx=(10, 10))
tk.Label(master, 
         text="Webcam").grid(row=2,padx=(10, 10))

tk.Label(master, 
         text="Model",font='Helvetica 12 bold').grid(row=4,padx=(10, 10),pady=10)

tk.Label(master, 
         text="Alternative model (Faster RCNN)").grid(row=5,padx=(10, 10))

tk.Label(master, 
         text="Skipped frames").grid(row=6,padx=(10, 10))

tk.Label(master, 
         text="Logging",font='Helvetica 12 bold').grid(row=7,padx=(10, 10),pady=10)

tk.Label(master, 
         text="Seperate session logging files").grid(row=8,padx=(10, 10),pady=0)

tk.Label(master, 
         text="Detection",font='Helvetica 12 bold').grid(row=9,padx=(10, 10),pady=10)

tk.Label(master, 
         text="Detection Confidence (0-1)").grid(row=10,padx=(10, 10),pady=0)

tk.Label(master, 
         text="Zone detection time").grid(row=11,padx=(10, 10),pady=0)


def webcam_checkbox_change():
    if webcam.get():
        disableEntry()
    else:
        enableEntry()

def enableEntry():
    file_entry.configure(state="normal")
    file_entry.update()
    button_explore.configure(state="normal")
    button_explore.update()

def disableEntry():
    file_entry.configure(state="disabled")
    file_entry.update()
    button_explore.configure(state="disabled")
    button_explore.update()


webcam_checkbox = tk.Checkbutton(master,variable=webcam, onvalue=True, offvalue=False,command=webcam_checkbox_change)
skipped_frames_entry = tk.Entry(master)
skipped_frames_entry.insert(0,skipped_frames)
model_checkbox = tk.Checkbutton(master,variable=model,onvalue=True,offvalue=False)
seperate_session_logging_files_checkbox = tk.Checkbutton(master,variable=seperate_session_logging_files,onvalue=True,offvalue=False)
detection_confidence_entry = tk.Entry(master)
detection_confidence_entry.insert(0,detection_confidence)
zone_detection_time_entry = tk.Entry(master)
zone_detection_time_entry.insert(0,zone_detection_time)

if webcam.get():
    webcam_checkbox.select()
    webcam_checkbox_change()

if seperate_session_logging_files.get():
    seperate_session_logging_files_checkbox.select()

if model.get():
    model_checkbox.select()


file_entry.grid(row=1, column=1)
button_explore.grid(row=1,column=2, padx=(10, 10))
webcam_checkbox.grid(row=2, column=1)
model_checkbox.grid(row=5,column=1)
skipped_frames_entry.grid(row=6,column=1)
seperate_session_logging_files_checkbox.grid(row=8,column=1)
detection_confidence_entry.grid(row=10,column=1)
zone_detection_time_entry.grid(row=11,column=1)



def save():
    global webcam, file_entry, video_input_file,seperate_session_logging_files,skipped_frames,detection_confidence,zone_detection_time
    video_input_file = file_entry.get()
    skipped_frames = skipped_frames_entry.get()
    detection_confidence = detection_confidence_entry.get()
    zone_detection_time = zone_detection_time_entry.get()

    if float(zone_detection_time) < 0:
        zone_detection_time = 0
    if float(detection_confidence) < 0:
        detection_confidence = 0
    if float(detection_confidence) > 1:
        detection_confidence = 1
    if int(skipped_frames) < 1:
        skipped_frames = 1

    config_object["VIDEOSTREAM"] = {
        "video_input": video_input_file,
        "webcam": webcam.get(),
    }

    config_object["MODEL"] = {
        "alt_model_faster_rcnn" : model.get(),
        "skipped_frames": skipped_frames,
    }

    config_object["LOGGING"] = {
        "seperate_session_logging_files" : seperate_session_logging_files.get()
    }

    config_object["DETECTION"] = {
        "detection_confidence" : detection_confidence,
        "zone_detection_time": zone_detection_time,
    }

    with open('config.ini', 'w') as conf:
        config_object.write(conf)

def reset():
    webcam_checkbox.select()
    file_entry.delete(0,'end')
    model_checkbox.deselect()
    skipped_frames_entry.delete(0,'end')
    skipped_frames_entry.insert(0,2)
    seperate_session_logging_files_checkbox.select()
    detection_confidence_entry.delete(0,'end')
    detection_confidence_entry.insert(0,0.4)
    zone_detection_time_entry.delete(0,'end')
    zone_detection_time_entry.insert(0,2.5)
    save()

def run():
    save()
    master.destroy()
    input = ["-i",video_input_file]
    skip_frames = ["-s",skipped_frames]
    det_conf = ["-c",detection_confidence]
    zone_det = ["-t",zone_detection_time]
    call = ['python3', 'people_counter.py'] + det_conf + skip_frames + zone_det
    
    if not webcam.get():
        call = call + input
    if model.get():
        call = call + ["-m"]
        
    print(call)
    subprocess.call(call)




tk.Button(master, 
          text='Save', 
          command=save).grid(row=12, 
                                    column=0, 
                                    sticky=tk.W, 
                                    pady=(20,10),padx=10)
tk.Button(master, 
          text='Reset', command=reset).grid(row=12, 
                                                       column=1, 
                                                       sticky=tk.W, 
                                                       pady=(20,10),padx=10)

tk.Button(master, 
          text='Run', command=run).grid(row=12, 
                                                       column=2, 
                                                       sticky=tk.W, 
                                                       pady=(20,10),padx=10)

tk.mainloop()
