import cv2
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO

# ---------------------- RECOMMENDATION BUCKETS (GENERIC) ----------------------
recommendation_buckets = {
    # Snacks & Chips
    'Kolson Slanty Jalapeno': ['Lays Masala', 'Super Crisp BBQ', 'Kurkure Chutney Chaska'],
    'Lays Masala': ['Kolson Slanty Jalapeno', 'Kurkure Chutney Chaska', 'Lays Wavy Mexican Chili'],
    'Lays Wavy Mexican Chili': ['Super Crisp BBQ', 'Kolson Slanty Jalapeno'],
    'Kurkure Chutney Chaska': ['Lays Masala', 'Super Crisp BBQ'],
    'Super Crisp BBQ': ['Lays Masala', 'Kurkure Chutney Chaska'],

    # Biscuits & Cookies
    'LU Candi Biscuit': ['LU Prince Biscuit', 'Peek Freans Sooper Biscuit'],
    'LU Prince Biscuit': ['LU Oreo Biscuit', 'Peek Freans Sooper Biscuit'],
    'Peek Freans Sooper Biscuit': ['LU Oreo Biscuit', 'Bisconni Chocolate Chip Cookies'],
    'LU Oreo Biscuit': ['LU Candi Biscuit', 'Bisconni Chocolate Chip Cookies'],
    'Bisconni Chocolate Chip Cookies': ['LU Prince Biscuit', 'Peek Freans Sooper Biscuit'],

    # Soft Drinks & Juices
    'Coca Cola Can': ['Fanta', 'Nestle Fruita Vitals Red Grapes', 'Shezan Apple'],
    'Fanta': ['Coca Cola Can', 'Shezan Apple', 'Fresher Guava Nectar'],
    'Nestle Fruita Vitals Red Grapes': ['Shezan Apple', 'Fresher Guava Nectar'],
    'Shezan Apple': ['Fanta', 'Nestle Fruita Vitals Red Grapes'],
    'Fresher Guava Nectar': ['Shezan Apple', 'Nestle Fruita Vitals Red Grapes'],

    # Tea & Related
    'Islamabad Tea': ['Tapal Danedar', 'Supreme Tea', 'Meezan Ultra Rich Tea'],
    'Lipton Yellow Label Tea': ['Tapal Danedar', 'Meezan Ultra Rich Tea'],
    'Meezan Ultra Rich Tea': ['Supreme Tea', 'Tapal Danedar'],
    'Supreme Tea': ['Islamabad Tea', 'Lipton Yellow Label Tea'],
    'Tapal Danedar': ['Meezan Ultra Rich Tea', 'Lipton Yellow Label Tea'],

    # Personal Care
    'Lifebuoy Total Protect Soap': ['Safeguard Bar Soap Pure White', 'Vaseline Healthy White Lotion'],
    'Safeguard Bar Soap Pure White': ['Lifebuoy Total Protect Soap', 'Sunsilk Shampoo Soft & Smooth'],
    'Sunsilk Shampoo Soft & Smooth': ['Vaseline Healthy White Lotion', 'Safeguard Bar Soap Pure White'],
    'Vaseline Healthy White Lotion': ['Sunsilk Shampoo Soft & Smooth', 'Lifebuoy Total Protect Soap'],

    # Toothpaste
    'Colgate Maximum Cavity Protection': ['Safeguard Bar Soap Pure White', 'Lifebuoy Total Protect Soap'],
}

# ---------------------- RECOMMENDATION FUNCTION ----------------------
def get_recommendations(detections, model_names, buckets):
    detected_items = set([model_names[int(cls)] for cls in detections])
    recommended = set()
    for item in detected_items:
        if item in buckets:
            recommended.update(buckets[item])
    return recommended - detected_items

# ---------------------- MAIN GUI CLASS ----------------------
class YOLOCameraApp:
    def __init__(self, window, model_path='best.pt'):
        self.window = window
        self.window.title("🛒 Grocery Detector with Recommendations")
        self.window.geometry("950x700")
        self.window.configure(bg="#f5f5f5")

        self.video_stream = None
        self.running = False
        self.cart = []
        self.latest_detections = []

        self.model = YOLO(model_path)

        style = ttk.Style()
        style.configure("TButton", font=("Segoe UI", 10), padding=6)

        self.label = tk.Label(window, bg="#cccccc")
        self.label.pack(padx=10, pady=10)

        button_frame = tk.Frame(window, bg="#f5f5f5")
        button_frame.pack(pady=10)

        self.start_button = ttk.Button(button_frame, text="▶ Start", command=self.start_camera)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(button_frame, text="■ Stop", command=self.stop_camera)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        self.add_button = ttk.Button(button_frame, text="＋ Add to Cart", command=self.add_to_cart)
        self.add_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = ttk.Button(button_frame, text="⛔ Clear Cart", command=self.clear_cart)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.recommend_label = tk.Label(window, text="Recommendations:", font=("Segoe UI", 12, "bold"), fg="#004080", bg="#f5f5f5")
        self.recommend_label.pack(pady=(10, 0), anchor="w", padx=20)

        self.cart_label = tk.Label(window, text="Cart:", font=("Segoe UI", 12, "bold"), fg="#006600", bg="#f5f5f5", anchor="w", justify="left")
        self.cart_label.pack(pady=5, fill=tk.X, padx=20)

    def start_camera(self):
        if not self.running:
            self.running = True
            self.video_stream = cv2.VideoCapture(0)
            self.thread = threading.Thread(target=self.update_frame)
            self.thread.start()

    def stop_camera(self):
        self.running = False
        if self.video_stream is not None:
            self.video_stream.release()
        self.label.config(image='')

    def add_to_cart(self):
        for item in self.latest_detections:
            if item not in self.cart:
                self.cart.append(item)
        self.update_cart_label()

    def clear_cart(self):
        self.cart = []
        self.update_cart_label()

    def update_cart_label(self):
        self.cart_label.config(text="Cart: " + ', '.join(self.cart) if self.cart else "Cart is empty")

    def update_frame(self):
        while self.running:
            ret, frame = self.video_stream.read()
            if not ret:
                break

            results = self.model.predict(source=frame, conf=0.3, save=False)
            boxes = results[0].boxes.cls.tolist() if results[0].boxes else []
            self.latest_detections = [self.model.names[int(cls)] for cls in boxes]

            recommended_items = get_recommendations(boxes, self.model.names, recommendation_buckets)
            self.recommend_label.config(text="Recommendations: " + ', '.join(recommended_items) if recommended_items else "Recommendations: None")

            annotated_frame = results[0].plot()
            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_image)
            imgtk = ImageTk.PhotoImage(image=img)

            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

        if self.video_stream:
            self.video_stream.release()

# ---------------------- RUN APP ----------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOCameraApp(root, model_path=r'/workspaces/group-projects-ab39912/final-report/final_application/best.pt')
    root.mainloop()
