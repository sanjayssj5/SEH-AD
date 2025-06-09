
# Docker Setup and Usage Guide for the Application

> 🚀 **Docker Repository**: [docker pull sanjayssj4/seh-ad:app]

This GitHub repository includes a `Dockerfile-DockerCUDA` to build the Docker image directly. However, note that **the model and sample images for prediction are not included**.

---

## 📁 Folder Structure Setup

Before running the Docker image:

1. Create a folder named `imgs` with the following subfolders:
   - `capture`
   - `acquisitions`

---

## 🔌 Realsense Camera Setup (Windows OS)

Ensure Realsense cameras are connected to USB ports.

To access USB devices in Docker via WSL (Windows Subsystem for Linux), follow these steps:

1. **Install `usbipd`**:  
   Run either of the following:
   ```bash
   winget install --interactive --exact dorssel.usbipd-win
   ```
   or download the `.msi` installer from [Microsoft Docs](https://learn.microsoft.com/en-us/windows/wsl/connect-usb).

2. **List connected USB devices**:
   ```bash
   usbipd list
   ```

3. **Bind the Realsense device** (replace `2-13` with your actual bus ID):
   ```bash
   usbipd bind --busid 2-13
   ```

4. **Attach the device to WSL**:
   ```bash
   usbipd attach --wsl --busid 2-13
   ```

---

## 🐳 Run the Docker Container

> 🛑 Run as **administrator** on Windows or with **sudo** on Linux.

Mount the `imgs` folder so outputs from acquisitions and predictions are visible.

```bash
docker run -it --rm \
  -v /dev:/dev \
  -v your\path\to\imgs\in\host:/home/experiments/imgs \
  --device-cgroup-rule "c 81:* rmw" \
  --device-cgroup-rule "c 189:* rmw" \
  sanjayssj4/seh-ad:app
```

---

## ✅ Verifying Inside the Container

1. **Check USB devices**:
   ```bash
   lsusb
   ```

2. **Check Realsense cameras**:
   ```bash
   rs-enumerate-devices
   ```

---

## 📸 Applications Inside the Container

### 1. **Image Acquisition** (Dual/Single Camera)

Run:
```bash
python3 acquisition.py
```

**How it works:**

- First, the script asks if you know the left/right camera positions.
  - **If no**: It captures images from both cameras and exits.
  - Review images in the `imgs` folder and rerun.
  - **If yes**: Enter the serial number for left/right cameras. The program saves a `camera_mapping.json` file.

- **Image Capture Mode**:
  - Press **`p` + `Enter`**: Capture and name the image.
  - Press **`q` + `Enter`**: Exit the application.

- **To reconfigure camera position**:  Delete the  `camera_mapping.json` file.
---

### 2. **Image Acquisition & Prediction** (Single Camera)

Run:
```bash
python3 experiment.py
```

**How it works:**

- Disconnect other cameras—only one should be active.
- Runs `simple_capture.py` to acquire an image, saved in `imgs/capture`.
- If successful, it calls `predict_patched_per_partnorm.py` to predict anomalies.
- Results saved in the `predicted` folder.

