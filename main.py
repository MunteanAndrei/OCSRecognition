from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2
from ultralytics import YOLO
from color_and_shape_detection import detect_color, detect_shape

class YOLOApp(App):
    def build(self):
        self.capture = cv2.VideoCapture(1)  # Change the number 0-4 (maybe 1 or 2 is pc and 3 or 4 droid cam)
        self.model = YOLO('yolov8n.pt')
        layout = BoxLayout()
        self.img_widget = Image()
        layout.add_widget(self.img_widget)
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # 30 FPS
        return layout

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            results = self.model(frame)
            for result in results:
                processed_frame = frame.copy()
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = int(box.cls)
                    confidence = float(box.conf)  
                    label_name = self.model.names[label]

                    if label == 0:  # YOLOv8 utilizează de obicei `0` pentru persoană
                        width = x2 - x1
                        height = y2 - y1
                        scale_factor = 0.8
                        new_width = int(width * scale_factor)
                        new_height = int(height * scale_factor)
                        x1 = x1 + (width - new_width) // 2
                        y1 = y1 + (height - new_height) // 2
                        x2 = x1 + new_width
                        y2 = y1 + new_height
                        cv2.putText(processed_frame, f"{label_name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Schimbat în albastru
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(processed_frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    else:
                        color = detect_color(frame, (x1, y1, x2, y2))
                        shape = detect_shape(frame, (x1, y1, x2, y2))
                        cv2.putText(processed_frame, f"{label_name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)  # Adăugat eticheta obiectului
                        cv2.putText(processed_frame, f"Color: {color}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)  # Schimbat în galben și mutat mai sus
                        cv2.putText(processed_frame, f"Shape: {shape}", (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)  # Schimbat în galben și mutat mai sus
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Adăugat dreptunghi pentru alte obiecte

            buf1 = cv2.flip(processed_frame, 0)
            buf = buf1.tobytes()
            image_texture = Texture.create(size=(processed_frame.shape[1], processed_frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img_widget.texture = image_texture

    def on_stop(self):
        self.capture.release()

if __name__ == '__main__':
    YOLOApp().run()
