import pyrealsense2 as rs
import numpy as np
import cv2
import os
import json
from datetime import datetime

# Configuration
CONFIG_FILE = "config_default.json"
IMAGE_DIR = "imgs/capture"

def load_config():
    """Load configuration from JSON file"""
    with open(CONFIG_FILE) as f:
        return json.load(f)

def capture_single_image():
    """Capture a single image from the first available RealSense camera"""
    try:
        # Load configuration
        config_data = load_config()
        json_string = str(config_data).replace("'", '"')
        
        # Initialize camera
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if not devices:
            print("No RealSense devices found")
            return False

        # Use first device
        serial = devices[0].get_info(rs.camera_info.serial_number)
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        
        # Configure stream from JSON config
        stream_config = config_data['viewer']
        config.enable_stream(
            rs.stream.color,
            int(stream_config['stream-width']),
            int(stream_config['stream-height']),
            rs.format.bgr8,
            int(stream_config['stream-fps'])
        )

        # Start pipeline
        cfg = pipeline.start(config)
        device = cfg.get_device()
        
        # Apply advanced mode settings
        advnc_mode = rs.rs400_advanced_mode(device)
        advnc_mode.load_json(json_string)


         # Warmup - capture and discard several frames
        for _ in range(30):
            pipeline.wait_for_frames()
        # Capture frame
        try:
            frames = pipeline.wait_for_frames(6000)  # 6 second timeout
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                print("No color frame received")
                return False
                
            # Save image
            os.makedirs(IMAGE_DIR, exist_ok=True)
            image = np.asanyarray(color_frame.get_data())
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"sample_{timestamp}.png"
            cv2.imwrite(os.path.join(IMAGE_DIR, filename), image)
            print(f"Image saved to {os.path.join(IMAGE_DIR, filename)}")
            return True
            
        except Exception as e:
            print(f"Capture error: {e}")
            device.hardware_reset()
            return False
            
    except Exception as e:
        print(f"Initialization error: {e}")
        return False
        
    finally:
        try:
            pipeline.stop()
        except:
            pass

if __name__ == "__main__":
    success = capture_single_image()
    print(f"Capture successful: {success}")
