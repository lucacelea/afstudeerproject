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

else:
    webcam = tk.BooleanVar()
    webcam.set(True)
    video_input_file = ""
    skipped_frames = 1
    seperate_session_logging_files = tk.BooleanVar()
    seperate_session_logging_files.set(True)



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
         text="Skipped frames").grid(row=5,padx=(10, 10))

tk.Label(master, 
         text="Logging",font='Helvetica 12 bold').grid(row=6,padx=(10, 10),pady=10)

tk.Label(master, 
         text="Seperate session logging files").grid(row=7,padx=(10, 10),pady=0)


def webcam_checkbox_change():
    if webcam.get():
        disableEntry()
    else:
        enableEntry()

def enableEntry():
    file_entry.configure(state="normal")
    file_entry.update()
    button_explore.configure(state="normalgit")
    button_explore.update()

def disableEntry():
    file_entry.configure(state="disabled")
    file_entry.update()
    button_explore.configure(state="disabled")
    button_explore.update()


webcam_checkbox = tk.Checkbutton(master,variable=webcam, onvalue=True, offvalue=False,command=webcam_checkbox_change)
skipped_frames_entry = tk.Entry(master)
skipped_frames_entry.insert(0,skipped_frames)
seperate_session_logging_files_checkbox = tk.Checkbutton(master,variable=seperate_session_logging_files,onvalue=True,offvalue=False)

if webcam.get():
    webcam_checkbox.select()
    webcam_checkbox_change()

if seperate_session_logging_files.get():
    seperate_session_logging_files_checkbox.select()


file_entry.grid(row=1, column=1)
button_explore.grid(row=1,column=2, padx=(10, 10))
webcam_checkbox.grid(row=2, column=1)
skipped_frames_entry.grid(row=5,column=1)
seperate_session_logging_files_checkbox.grid(row=7,column=1)


def save():
    global webcam, file_entry, video_input_file,seperate_session_logging_files
    video_input_file = file_entry.get()
    skipped_frames = skipped_frames_entry.get()

    config_object["VIDEOSTREAM"] = {
        "video_input": video_input_file,
        "webcam": webcam.get(),
    }

    config_object["MODEL"] = {
        "skipped_frames": skipped_frames,
    }

    config_object["LOGGING"] = {
        "seperate_session_logging_files" : seperate_session_logging_files.get()
    }

    with open('config.ini', 'w') as conf:
        config_object.write(conf)

def run():
    save()
    master.destroy()
    protox = ["-p","mobilenet_ssd/MobileNetSSD_deploy.prototxt"]
    model = ["-m","mobilenet_ssd/MobileNetSSD_deploy.caffemodel"]
    input = ["-i",video_input_file]
    skip_frames = ["-s",skipped_frames]

    if webcam.get():
        subprocess.call(['python3', 'people_counter.py'] + protox + model + skip_frames)
    else:
        subprocess.call(['python3', 'people_counter.py'] + protox + model + input + skip_frames)

def reset():
    webcam_checkbox.select()
    file_entry.delete(0,'end')
    skipped_frames_entry.delete(0,'end')
    skipped_frames_entry.insert(0,1)
    seperate_session_logging_files_checkbox.select()
    save()


tk.Button(master, 
          text='Save', 
          command=save).grid(row=8, 
                                    column=0, 
                                    sticky=tk.W, 
                                    pady=(20,10),padx=10)
tk.Button(master, 
          text='Reset', command=reset).grid(row=8, 
                                                       column=1, 
                                                       sticky=tk.W, 
                                                       pady=(20,10),padx=10)

tk.Button(master, 
          text='Run', command=run).grid(row=8, 
                                                       column=2, 
                                                       sticky=tk.W, 
                                                       pady=(20,10),padx=10)

tk.mainloop()
