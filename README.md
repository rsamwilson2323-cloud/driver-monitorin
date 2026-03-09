# 🚗 Driver Monitoring System (Computer Vision)

Driver Monitoring System is a real-time safety system built using **Python 🐍, OpenCV 🎥, MediaPipe 🧠, and YOLOv8 🤖**.

The system monitors driver behavior through a webcam to detect **drowsiness, attention level, mobile phone usage, and seat belt usage**. It provides live alerts to help improve road safety.

---

# ✨ Features

👁 **Eye State Detection**

* Detects **eye open / eye closed**
* Calculates **Eye Aspect Ratio (EAR)**

😴 **Drowsiness Detection**

* Triggers alert if eyes remain closed for more than **2 seconds**

👀 **Driver Attention Tracking**

* Detects whether the driver is **looking forward or looking away**

📱 **Mobile Phone Detection**

* Uses **YOLOv8 object detection** to detect phone usage while driving

🪢 **Seat Belt Detection**

* Heuristic-based detection using **edge detection and Hough lines**

🖥 **Live Monitoring Dashboard**

* Displays real-time driver status and safety alerts

---

# 📂 Project Structure

```
driver-monitoring-system
│
├── driver_monitoring.py
├── requirements.txt
├── README.md
└── LICENSE
```

The **YOLOv8 model (yolov8n.pt)** will automatically download during the first run if not present.

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```
git clone https://github.com/rsamwilson2323-cloud/driver-monitorin.git
cd driver-monitorin
```

---

## 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

# 📦 Requirements

Main libraries used:

```
opencv-python
mediapipe
numpy
ultralytics
```

---

# ▶️ Usage

Run the system using:

```
python driver_monitoring.py
```

📷 The webcam will start automatically.

To stop the program:

**Press ENTER ⏎**

---

# 🧠 How It Works

👁 **Eye Aspect Ratio (EAR)**
Uses facial landmarks from **MediaPipe Face Mesh** to detect eye closure and identify drowsiness.

👀 **Head Position (Attention)**
Tracks nose tip movement to determine whether the driver is **looking forward or away from the road**.

📱 **Object Detection (YOLOv8)**
Detects **mobile phone usage** in real time using YOLOv8.

🪢 **Seat Belt Detection**
Uses **Canny Edge Detection and Hough Transform** to detect diagonal lines across the chest area representing a seat belt.

---

# ⚠️ Alerts Displayed

The system may display alerts such as:

* **EYE CLOSED**
* **DROWSINESS ALERT**
* **LOOKING AWAY**
* **PHONE USAGE: YES**
* **SEAT BELT: NOT WORN**

---

# 📸 Example Output

```
Eye: EYE OPEN
Attention: LOOKING FORWARD
Seat Belt: WORN
Phone Usage: NO
```

---

# 🚀 Future Improvements

🔊 Sound alert for drowsiness
📊 Driver behavior logging
🧠 Deep learning-based seat belt detection
🚘 Integration with vehicle safety systems
📱 Mobile or embedded system deployment

---

# ⚠️ Disclaimer

This project is intended for **educational and research purposes only**.
It should not be used as a replacement for professional driver safety systems.

---

# 👨‍💻 Author

**Sam Wilson**

🌐 GitHub
https://github.com/rsamwilson2323-cloud

💼 LinkedIn
https://www.linkedin.com/in/sam-wilson-14b554385

---

# 📜 License

This project is licensed under the **MIT License**.
