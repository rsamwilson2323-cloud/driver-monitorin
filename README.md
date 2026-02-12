🚗 Driver Monitoring System (Computer Vision)

A real-time Driver Monitoring System built using Python, OpenCV, MediaPipe, and YOLOv8 to enhance road safety.
This system monitors driver drowsiness, attention, mobile phone usage, and seat belt usage through a webcam.


🔍 Features

👁 Eye State Detection

Detects eye open / eye closed

Calculates Eye Aspect Ratio (EAR)

😴 Drowsiness Detection

Triggers alert if eyes remain closed for more than 2 seconds

👀 Driver Attention Tracking

Detects whether the driver is looking forward or looking away

📱 Mobile Phone Detection

Uses YOLOv8 to detect phone usage while driving

🪢 Seat Belt Detection

Heuristic-based detection using edge detection & Hough lines

🖥 Live Monitoring Dashboard

Displays real-time alerts and driver status on screen

🛠 Technologies Used

Python 3

OpenCV

MediaPipe (Face Mesh)

YOLOv8 (Ultralytics)

NumPy

📦 Installation
1️⃣ Clone the Repository
git clone https://github.com/your-username/driver-monitoring-system.git
cd driver-monitoring-system

2️⃣ Install Required Packages
pip install opencv-python mediapipe numpy ultralytics

3️⃣ Download YOLOv8 Model

The code uses:

yolov8n.pt


It will auto-download on first run, or you can manually place it in the project folder.

▶️ How to Run
python driver_monitoring.py


Webcam will start automatically

Press ENTER to exit the application

🧠 How It Works
👁 Eye Aspect Ratio (EAR)

Uses facial landmarks from MediaPipe

Detects eye closure duration to identify drowsiness

👃 Head Position (Attention)

Tracks nose tip movement

Determines if driver is looking away from the road

📱 Object Detection (YOLOv8)

Detects cell phone usage in real time

🪢 Seat Belt Detection

Detects diagonal lines across chest region

Uses Canny Edge Detection + Hough Transform

⚠️ Alerts Displayed

EYE CLOSED

DROWSINESS ALERT

LOOKING AWAY

PHONE USAGE: YES

SEAT BELT: NOT WORN

📸 Output Example
Eye: EYE OPEN
Attention: LOOKING FORWARD
Seat Belt: WORN
Phone Usage: NO

🚀 Future Improvements

🔊 Sound alert for drowsiness

📊 Driver behavior logging

🧠 Deep learning-based seat belt detection

🚘 Integration with vehicle systems

📱 Mobile / Embedded deployment

⚖️ Disclaimer

This project is for educational and research purposes only.
It should not be used as a replacement for professional driver safety systems.

👨‍💻 Author

Sam Wilson
📧 Email: rsamwilson2323@gmail.com
🖇️ LinkedIn: https://www.linkedin.com/in/sam-wilson-14b554385
🔗 GitHub: https://github.com/rsamwilson2323-cloud
