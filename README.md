# varicose-vein-detection
Detection and Severity Grading of Varicose Vein
(Mild vs Moderate Classification)

Project Overview

This mini-project detects the presence of varicose veins in leg images and classifies their severity as Mild or Moderate using deep learning models.
If no varicose vein is detected, the system outputs Normal.
The system uses:
YOLOv8 for varicose vein detection
EfficientNet-B0 (CNN) for severity grading

System Workflow

User uploads a leg image through the web interface.
YOLOv8 checks for varicose vein presence.
If detected:
Image is passed to severity classifier.
EfficientNet-B0 predicts severity:
Mild
Moderate
If no detection:
Output: Normal

Technologies Used:

Python
Flask (Web interface)
YOLOv8 (Ultralytics)
TensorFlow / Keras
EfficientNet-B0
