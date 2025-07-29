import subprocess
import sys
import os

import shutil

image_path = "imgs/capture"

def clear_all_in_directory(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def run_capture_script():
    """Run the camera capture script and return success status."""
    process = subprocess.Popen(
        ["python3", "simple_capture.py"],
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True
    )
    process.wait()
    return process.returncode == 0

def run_prediction_script():
    """Run the prediction script if capture was successful."""
    script_path = "predict_patched_per_partnorm.py"
    model_dir = "model_split_per_partnorm"

    process = subprocess.Popen(
        [
            "python3",
            script_path,
            "--model_dir", model_dir,
            "--image_path", image_path,
            # The threshold should be changed according to the trained model
            "--threshold", 0.4135,
        ],
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True
    )
    process.wait()
    return process.returncode == 0

def main():
    if run_capture_script():
        run_prediction_script()
        clear_all_in_directory(image_path)

if __name__ == "__main__":
    main()
