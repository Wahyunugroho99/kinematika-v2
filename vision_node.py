#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import cv2
from ultralytics import YOLO

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.publisher_ = self.create_publisher(Point, '/target_bbox', 10)
        self.timer = self.create_timer(0.1, self.timer_callback) # 10 FPS
        
        # Load YOLO Model (Pastikan path best.pt benar)
        self.model = YOLO('best.pt') 
        self.cap = cv2.VideoCapture(1)
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        
        self.get_logger().info("Vision Node (Eye-to-Hand) Started.")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret: return

        results = self.model(frame, conf=0.5, verbose=False)
        msg = Point()
        msg.x, msg.y, msg.z = -1.0, -1.0, 0.0 # Default: Tidak ada deteksi

        for r in results:
            boxes = r.boxes
            if len(boxes) > 0:
                # Ambil objek pertama yang terdeteksi
                box = boxes[0]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Normalisasi koordinat (0.0 - 1.0)
                msg.x = ((x1 + x2) / 2) / 640.0
                msg.y = ((y1 + y2) / 2) / 480.0
                msg.z = ((x2 - x1) * (y2 - y1)) / (640.0 * 480.0)
                break # Hanya fokus 1 objek

        self.publisher_.publish(msg)

        # Opsional: Tampilkan window kamera (bisa dimatikan jika bikin berat RPi)
        cv2.imshow("Kamera Eye-to-Hand", results[0].plot())
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()