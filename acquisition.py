import pyrealsense2 as rs
import numpy as np
import cv2
import os
import json
import threading
import sys
import select
from datetime import datetime

# Paths
CONFIG_FILE = "config_default.json"
MAPPING_FILE = "camera_mapping.json"
IMAGE_DIR = "imgs/acquisitions"

# Clean old images
#for f in os.listdir(IMAGE_DIR):
#    if f.endswith(".png"):
#        os.remove(os.path.join(IMAGE_DIR, f))

# Load config
with open(CONFIG_FILE) as f:
    jsonObj = json.load(f)
json_string = str(jsonObj).replace("'", '\"')

# Detect cameras
ctx = rs.context()
serials = [dev.get_info(rs.camera_info.serial_number) for dev in ctx.devices]


def save_mapping(mapping):
    with open(MAPPING_FILE, "w") as f:
        json.dump(mapping, f)

def load_mapping():
    if os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE) as f:
            return json.load(f)
    return {}

capture_event = threading.Event()
exit_event = False
label_input = ""
cam_mapping = {}

def pipeline_health_check(pipeline):
    try:
        frames = pipeline.wait_for_frames(3000)
        return True
    except:
        return False

def run_camera(serial, label=None, stream_mode=False):
    global capture_event, exit_event, label_input

    config = rs.config()
    pipeline = rs.pipeline()
    config.enable_device(serial)
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    #print(device)
    if not any(s.get_info(rs.camera_info.name) == 'RGB Camera' for s in device.sensors):
        print(f"Camera {serial} has no RGB sensor. Skipping.")
        return

    config.enable_stream(
        rs.stream.color,
        int(jsonObj['viewer']['stream-width']),
        int(jsonObj['viewer']['stream-height']),
        rs.format.bgr8,
        int(jsonObj['viewer']['stream-fps'])
    )

    try:
        cfg = pipeline.start(config)
    except Exception as e:
        print(f"Error starting pipeline for {serial}: {e}. try plugging in again!!")
        exit_event = True

    dev = cfg.get_device()
    #print(dev)
    advnc_mode = rs.rs400_advanced_mode(dev)
    advnc_mode.load_json(json_string)

    try:
        if not stream_mode:
            print("Pipeline response in 3 seconds?:", pipeline_health_check(pipeline))
            color = pipeline.wait_for_frames().get_color_frame()
            img = np.asanyarray(color.get_data())
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"sample_{label or serial}_{timestamp}.png"
            cv2.imwrite(os.path.join(IMAGE_DIR, filename), img)
            print(f"[{serial}] Snapshot saved: {filename}")
        else:
            while not exit_event:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                color_image = np.asanyarray(color_frame.get_data())

                if capture_event.is_set():
                    label = cam_mapping.get(serial, serial)
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    filename = f"sample_{label}_{label_input}_{timestamp}.png"
                    cv2.imwrite(os.path.join(IMAGE_DIR, filename), color_image)
                    print(f"[{serial}] Captured: {filename}")
                    capture_event.clear()
    except Exception as e:
        print(f"Error capturing image for {serial}: {e} . Device disconnected try plugging in and running program again!!")
        dev.hardware_reset()
        exit_event = True

    finally:
        pipeline.stop()
        print(f"[{serial}] Stopped")


def input_listener():
    global capture_event, exit_event, label_input
    print("Press 'p' + [Enter] to capture, 'q' + [Enter] to quit.")
    while not exit_event:
        if select.select([sys.stdin], [], [], 0.1)[0]:
            cmd = sys.stdin.readline().strip()
            if cmd == 'p':
                label_input = input("Enter label for captured images: ").strip()
                capture_event.set() 
            elif cmd == 'q':
                exit_event = True
                break


# Interactive mapping
def interactive_mapping():
    global cam_mapping
    print(f"Detected cameras: {serials}")
    ans = input("Do you already know camera directions? (y/n): ").strip().lower()
    if ans == 'y':
        for sn in serials:
            direction = input(f"Enter direction for camera {sn} (e.g. left/right): ").strip().lower()
            cam_mapping[sn] = direction
        with open(MAPPING_FILE, 'w') as f:
            json.dump(cam_mapping, f)
    else:
        print("Capturing images to help you identify cameras...")
        for sn in serials:
            run_camera(sn, stream_mode=False)
        print(f"Images saved in {IMAGE_DIR}. Please check and re-run the program.")

# Load or initiate mapping
if os.path.exists(MAPPING_FILE):
    with open(MAPPING_FILE) as f:
        cam_mapping = json.load(f)
else:
    interactive_mapping()
    if not os.path.exists(MAPPING_FILE):
        exit()

# Stream mode if mapping known
threads = []
for sn in serials:
    label = cam_mapping.get(sn, sn)
    t = threading.Thread(target=run_camera, args=(sn, label, True))
    t.start()
    threads.append(t)

input_thread = threading.Thread(target=input_listener)
input_thread.start()

input_thread.join()


# Wait for all camera threads to finish
for t in threads:
    t.join()
