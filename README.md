# Object Detection and Recommendation for Retail

This is a real-time object detection application that uses a trained YOLOv8 model (`best.pt`) and activates your laptop's webcam to detect objects. The application is built with a simple graphical user interface (GUI) using `tkinter`.

## Files

- `app.py` – Python script that runs the application, loads the YOLO model, opens a GUI, and activates the webcam.
- `best.pt` – The trained YOLOv8 model file for object detection. Place this in the same directory as `app.py`.

## Requirements

You need Python 3.8 or later. Install the required libraries using pip:

```bash
pip install ultralytics opencv-python pillow

# To run the app
```bash
python app.py
