#!/usr/bin/env python3

import json
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO


def parse_camera_source(value):
    if isinstance(value, int):
        return value

    text = str(value).strip()
    if text.lstrip("-").isdigit():
        return int(text)

    return text


class YoloEyeToHandCameraNode(Node):
    def __init__(self):
        super().__init__("camera_yolo_eye_to_hand_node")

        dynamic_param = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter("model_path", "best.pt")
        self.declare_parameter("camera_source", "2", dynamic_param)
        self.declare_parameter("confidence", 0.8)
        self.declare_parameter("publish_hz", 15.0)

        self.model_path = str(self.get_parameter("model_path").value)
        self.camera_source = parse_camera_source(
            self.get_parameter("camera_source").value
        )
        self.confidence = float(self.get_parameter("confidence").value)
        self.publish_hz = max(1.0, float(self.get_parameter("publish_hz").value))

        self.bridge = CvBridge()
        self.model = None
        self.capture = None
        self.last_status_text = None
        self.last_status_time = 0.0
        self.last_camera_retry_time = 0.0
        self.camera_retry_interval = 2.0

        self.image_pub = self.create_publisher(
            Image,
            "/vision/image_annotated",
            10,
        )
        self.detections_pub = self.create_publisher(
            String,
            "/vision/detections",
            10,
        )
        self.status_pub = self.create_publisher(
            String,
            "/vision/status",
            10,
        )

        self._load_model()
        self._open_camera()

        self.timer = self.create_timer(1.0 / self.publish_hz, self._timer_callback)
        self.get_logger().info("CAMERA YOLO EYE-TO-HAND NODE AKTIF")

    def _publish_status(self, text, force=False):
        now = time.monotonic()
        if not force and text == self.last_status_text and now - self.last_status_time < 2.0:
            return

        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

        if text != self.last_status_text:
            self.get_logger().info(text)

        self.last_status_text = text
        self.last_status_time = now

    def _load_model(self):
        try:
            self.model = YOLO(self.model_path)
            self._publish_status(f"MODEL OK | {self.model_path}", force=True)
        except Exception as exc:
            self.model = None
            self._publish_status(f"ERROR MODEL | {exc}", force=True)

    def _open_camera(self):
        self.last_camera_retry_time = time.monotonic()
        self._release_camera()

        self.capture = cv2.VideoCapture(self.camera_source)
        if not self.capture.isOpened():
            self._release_camera()
            self._publish_status(
                f"ERROR CAMERA | tidak bisa membuka source {self.camera_source}",
                force=True,
            )
            return

        self._publish_status(f"CAMERA OK | source={self.camera_source}", force=True)

    def _release_camera(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def _timer_callback(self):
        if self.capture is None or not self.capture.isOpened():
            now = time.monotonic()
            if now - self.last_camera_retry_time >= self.camera_retry_interval:
                self._open_camera()
            return

        ok, frame = self.capture.read()
        if not ok or frame is None:
            self._release_camera()
            self._publish_status("ERROR CAMERA | gagal membaca frame", force=True)
            return

        annotated = frame
        detections = []
        status_text = None
        if self.model is not None:
            try:
                results = self.model.predict(
                    frame,
                    conf=self.confidence,
                    verbose=False,
                )
                result = results[0]
                annotated = result.plot()
                detections = self._extract_detections(result)
            except Exception as exc:
                status_text = f"ERROR YOLO | {exc}"
        else:
            status_text = "ERROR MODEL | YOLO belum siap"

        height, width = annotated.shape[:2]
        stamp = self.get_clock().now().to_msg()

        image_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        image_msg.header.stamp = stamp
        image_msg.header.frame_id = "eye_to_hand_camera"
        self.image_pub.publish(image_msg)

        packet = {
            "image_width": int(width),
            "image_height": int(height),
            "timestamp": time.time(),
            "detections": detections,
        }
        detections_msg = String()
        detections_msg.data = json.dumps(packet)
        self.detections_pub.publish(detections_msg)

        if status_text is None:
            status_text = f"CAMERA OK | {width}x{height} | {len(detections)} objek"
        self._publish_status(status_text)

    def _extract_detections(self, result):
        detections = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections

        for box in boxes:
            xyxy = box.xyxy[0].detach().cpu().tolist()
            confidence = float(box.conf[0].detach().cpu().item())
            class_id = int(box.cls[0].detach().cpu().item())
            x1, y1, x2, y2 = [float(value) for value in xyxy]
            detections.append(
                {
                    "class_id": class_id,
                    "label": self._class_label(class_id),
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                    "center": [
                        (x1 + x2) / 2.0,
                        (y1 + y2) / 2.0,
                    ],
                }
            )

        detections.sort(key=lambda item: item["confidence"], reverse=True)
        return detections

    def _class_label(self, class_id):
        names = getattr(self.model, "names", {})
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    def destroy_node(self):
        self._release_camera()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloEyeToHandCameraNode()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
