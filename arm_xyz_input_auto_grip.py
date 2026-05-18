#!/usr/bin/env python3

import json
import math
import os
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from PIL import Image as PilImage, ImageTk
from sensor_msgs.msg import Image as RosImage

from std_msgs.msg import Empty, Float64, Float64MultiArray, String


try:
    IMAGE_RESAMPLE = PilImage.Resampling.LANCZOS
except AttributeError:
    IMAGE_RESAMPLE = PilImage.LANCZOS


CALIBRATION_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "eye_to_hand_calibration.json",
)


# ============================================================
# ARM XYZ GUI NODE
#
# User input X Y Z dari GUI.
# Tidak ada input pitch.
#
# Publish:
# /arm/target_xyz = [x, y, z, duration]
# /arm/pick_drop_xyz = [pick_x, pick_y, pick_z, pick_duration,
#                       drop_x, drop_y, drop_z, drop_duration]
# /arm/gripper
# /arm/home
# /arm/stop
#
# Subscribe:
# /arm/current_xyz = [x, y, z, pitch]
# /arm/servo_angle = [base, shoulder, elbow, wrist, gripper]
# /arm/status
# /vision/image_annotated
# /vision/detections
# /vision/status
# ============================================================


def clamp(value, low, high):
    return max(low, min(high, value))


class ArmGeometry:
    L1 = 8.0
    L2 = 12.0
    L3 = 8.0
    L4 = 15.0


class JointCalibration:
    def __init__(
        self,
        model_home_deg,
        servo_home_deg=90.0,
        direction=1,
        trim_deg=0.0,
    ):
        self.model_home_deg = model_home_deg
        self.servo_home_deg = servo_home_deg
        self.direction = direction
        self.trim_deg = trim_deg

    def servo_to_model(self, servo_deg):
        return (
            self.model_home_deg
            + self.direction * (servo_deg - self.servo_home_deg - self.trim_deg)
        )


JOINT_CALIBRATION = {
    "base": JointCalibration(model_home_deg=0.0, direction=-1),
    "shoulder": JointCalibration(model_home_deg=90.0, direction=1),
    "elbow": JointCalibration(model_home_deg=-90.0, direction=-1),
    "wrist": JointCalibration(model_home_deg=0.0, direction=-1),
}


HOME_MODEL_ANGLES = {
    "base": 0.0,
    "shoulder": 90.0,
    "elbow": -90.0,
    "wrist": 0.0,
}


class RobotVisualizerKinematics:
    def __init__(self):
        self.geo = ArmGeometry()

    def joint_positions(self, q):
        q0 = math.radians(q["base"])
        q1 = math.radians(q["shoulder"])
        q2 = math.radians(q["elbow"])
        q3 = math.radians(q["wrist"])

        positions = []

        def xyz_from_rz(r, z):
            return {
                "x": r * math.sin(q0),
                "y": r * math.cos(q0),
                "z": z,
            }

        positions.append({"name": "base", "x": 0.0, "y": 0.0, "z": 0.0})
        positions.append({"name": "shoulder", "x": 0.0, "y": 0.0, "z": self.geo.L1})

        r = self.geo.L2 * math.cos(q1)
        z = self.geo.L1 + self.geo.L2 * math.sin(q1)
        positions.append({"name": "elbow", **xyz_from_rz(r, z)})

        r += self.geo.L3 * math.cos(q1 + q2)
        z += self.geo.L3 * math.sin(q1 + q2)
        positions.append({"name": "wrist", **xyz_from_rz(r, z)})

        r += self.geo.L4 * math.cos(q1 + q2 + q3)
        z += self.geo.L4 * math.sin(q1 + q2 + q3)
        positions.append({"name": "tip", **xyz_from_rz(r, z)})

        return positions


class ArmXYZGuiNode(Node):
    def __init__(self):
        super().__init__("arm_xyz_gui_node")
        self.bridge = CvBridge()

        self.pose_pub = self.create_publisher(
            Float64MultiArray,
            "/arm/target_xyz",
            10,
        )

        self.pick_drop_pub = self.create_publisher(
            Float64MultiArray,
            "/arm/pick_drop_xyz",
            10,
        )

        self.gripper_pub = self.create_publisher(
            Float64,
            "/arm/gripper",
            10,
        )

        self.home_pub = self.create_publisher(
            Empty,
            "/arm/home",
            10,
        )

        self.stop_pub = self.create_publisher(
            Empty,
            "/arm/stop",
            10,
        )

        self.current_xyz_sub = self.create_subscription(
            Float64MultiArray,
            "/arm/current_xyz",
            self.current_xyz_callback,
            10,
        )

        self.servo_angle_sub = self.create_subscription(
            Float64MultiArray,
            "/arm/servo_angle",
            self.servo_angle_callback,
            10,
        )

        self.status_sub = self.create_subscription(
            String,
            "/arm/status",
            self.status_callback,
            10,
        )

        self.camera_image_sub = self.create_subscription(
            RosImage,
            "/vision/image_annotated",
            self.camera_image_callback,
            10,
        )

        self.detections_sub = self.create_subscription(
            String,
            "/vision/detections",
            self.detections_callback,
            10,
        )

        self.vision_status_sub = self.create_subscription(
            String,
            "/vision/status",
            self.vision_status_callback,
            10,
        )

        self.current_xyz = {
            "x": 0.0,
            "y": 13.5,
            "z": 22.0,
            "pitch": 0.0,
        }
        self.model_angles = dict(HOME_MODEL_ANGLES)
        self.gripper_angle = 90.0
        self.status_text = "Menunggu data driver..."
        self.last_target = None
        self.last_drop = None
        self.last_error = ""
        self.camera_frame = None
        self.camera_frame_size = (0, 0)
        self.detections = []
        self.detections_seq = 0
        self.vision_status_text = "Menunggu data kamera..."

        self.get_logger().info("ARM XYZ GUI NODE AKTIF")

    def send_xyz(self, x, y, z, duration=2.0):
        self.last_error = ""
        msg = Float64MultiArray()
        msg.data = [
            float(x),
            float(y),
            float(z),
            float(duration),
        ]
        self.pose_pub.publish(msg)
        self.last_target = {
            "x": float(x),
            "y": float(y),
            "z": float(z),
        }
        self.status_text = (
            f"Target terkirim: X={x:.2f}, Y={y:.2f}, "
            f"Z={z:.2f}, Duration={duration:.2f}s"
        )

    def send_pick_drop(
        self,
        pick_x,
        pick_y,
        pick_z,
        pick_duration,
        drop_x,
        drop_y,
        drop_z,
        drop_duration,
    ):
        self.last_error = ""
        msg = Float64MultiArray()
        msg.data = [
            float(pick_x),
            float(pick_y),
            float(pick_z),
            float(pick_duration),
            float(drop_x),
            float(drop_y),
            float(drop_z),
            float(drop_duration),
        ]
        self.pick_drop_pub.publish(msg)
        self.last_target = {
            "x": float(pick_x),
            "y": float(pick_y),
            "z": float(pick_z),
        }
        self.last_drop = {
            "x": float(drop_x),
            "y": float(drop_y),
            "z": float(drop_z),
        }
        self.status_text = (
            f"Pick + Drop terkirim | "
            f"Pick X={pick_x:.2f}, Y={pick_y:.2f}, Z={pick_z:.2f} | "
            f"Drop X={drop_x:.2f}, Y={drop_y:.2f}, Z={drop_z:.2f}"
        )

    def send_gripper(self, angle):
        self.last_error = ""
        msg = Float64()
        msg.data = float(angle)
        self.gripper_pub.publish(msg)
        self.status_text = f"Gripper command: {angle:.2f} deg"

    def send_home(self):
        self.last_error = ""
        self.home_pub.publish(Empty())
        self.status_text = "Command HOME terkirim"

    def send_stop(self):
        self.last_error = ""
        self.stop_pub.publish(Empty())
        self.status_text = "Command STOP terkirim"

    def current_xyz_callback(self, msg):
        data = list(msg.data)
        if len(data) < 4:
            return

        self.current_xyz = {
            "x": float(data[0]),
            "y": float(data[1]),
            "z": float(data[2]),
            "pitch": float(data[3]),
        }

    def servo_angle_callback(self, msg):
        data = list(msg.data)
        if len(data) < 4:
            return

        servo_angles = {
            "base": float(data[0]),
            "shoulder": float(data[1]),
            "elbow": float(data[2]),
            "wrist": float(data[3]),
        }

        self.model_angles = {
            name: JOINT_CALIBRATION[name].servo_to_model(angle)
            for name, angle in servo_angles.items()
        }

        if len(data) >= 5:
            self.gripper_angle = float(data[4])

    def status_callback(self, msg):
        self.status_text = msg.data

    def camera_image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.vision_status_text = f"ERROR image kamera: {exc}"
            return

        self.camera_frame = frame
        height, width = frame.shape[:2]
        self.camera_frame_size = (width, height)

    def detections_callback(self, msg):
        try:
            packet = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.vision_status_text = f"ERROR JSON deteksi: {exc}"
            return

        detections = []
        for index, item in enumerate(packet.get("detections", [])):
            try:
                center = item.get("center", [0.0, 0.0])
                bbox = item.get("bbox", [0.0, 0.0, 0.0, 0.0])
                detections.append(
                    {
                        "index": index,
                        "class_id": int(item.get("class_id", -1)),
                        "label": str(item.get("label", "object")),
                        "confidence": float(item.get("confidence", 0.0)),
                        "bbox": [float(value) for value in bbox[:4]],
                        "center": [
                            float(center[0]),
                            float(center[1]),
                        ],
                    }
                )
            except (TypeError, ValueError, IndexError):
                continue

        detections.sort(key=lambda det: det["confidence"], reverse=True)
        self.detections = detections
        self.detections_seq += 1

        count = len(detections)
        self.vision_status_text = f"Deteksi YOLO: {count} objek"

    def vision_status_callback(self, msg):
        self.vision_status_text = msg.data


class ArmXYZGuiApp:
    def __init__(self, node):
        self.node = node
        self.kin = RobotVisualizerKinematics()

        self.root = tk.Tk()
        self.root.title("Robot Arm XYZ Control")
        self.root.geometry("1360x840")
        self.root.minsize(1180, 760)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.x_var = tk.StringVar(value="0")
        self.y_var = tk.StringVar(value="24")
        self.z_var = tk.StringVar(value="22")
        self.duration_var = tk.StringVar(value="2")
        self.drop_x_var = tk.StringVar(value="10")
        self.drop_y_var = tk.StringVar(value="24")
        self.drop_z_var = tk.StringVar(value="18")
        self.drop_duration_var = tk.StringVar(value="2")
        self.point_mode_var = tk.StringVar(value="target")
        self.pose_var = tk.StringVar(value="")
        self.servo_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Menunggu data driver...")
        self.vision_status_var = tk.StringVar(value="Vision: menunggu data kamera...")
        self.selected_object_var = tk.StringVar(value="Objek: belum ada")
        self.calibration_status_var = tk.StringVar(value="Kalibrasi: belum ada")
        self.calib_mode_var = tk.BooleanVar(value=False)
        self.calib_x_vars = [
            tk.StringVar(value=value)
            for value in ["-20", "20", "-20", "20"]
        ]
        self.calib_y_vars = [
            tk.StringVar(value=value)
            for value in ["10", "10", "40", "40"]
        ]

        self.camera_photo = None
        self.camera_display = None
        self.homography = None
        self.calibration_image_points = []
        self.calibration_robot_points = []
        self.visual_detections = []
        self.selected_detection_index = None
        self.last_detection_seq_drawn = -1
        self.last_calibration_version_drawn = -1
        self.calibration_version = 0

        self.node.last_drop = {
            "x": float(self.drop_x_var.get()),
            "y": float(self.drop_y_var.get()),
            "z": float(self.drop_z_var.get()),
        }
        for variable in [self.drop_x_var, self.drop_y_var, self.drop_z_var]:
            variable.trace_add("write", self._sync_drop_marker_from_inputs)
        self.z_var.trace_add("write", lambda *_args: self._update_selected_object_label())

        self._load_calibration()
        self._build_style()
        self._build_layout()
        self._update_labels()
        self._refresh_detection_table(force=True)
        self._draw_visuals()
        self._spin_ros()

    def _build_style(self):
        style = ttk.Style()
        style.configure("TFrame", background="#f4f6f8")
        style.configure("Panel.TFrame", background="#ffffff", relief="solid", borderwidth=1)
        style.configure("TLabel", background="#f4f6f8", foreground="#17202a")
        style.configure("Panel.TLabel", background="#ffffff", foreground="#17202a")
        style.configure("Title.TLabel", font=("TkDefaultFont", 13, "bold"))
        style.configure("Value.TLabel", font=("TkDefaultFont", 10))
        style.configure("Error.TLabel", foreground="#b42318")
        style.configure("Primary.TButton", padding=(10, 7))
        style.configure("Danger.TButton", padding=(10, 7))

    def _build_layout(self):
        self.root.configure(background="#f4f6f8")

        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(0, weight=0)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        control_panel = ttk.Frame(main, style="Panel.TFrame", padding=14)
        control_panel.grid(row=0, column=0, sticky="ns", padx=(0, 12))

        ttk.Label(
            control_panel,
            text="Input Pick + Drop",
            style="Panel.TLabel",
            font=("TkDefaultFont", 14, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))

        ttk.Label(
            control_panel,
            text="Target ambil",
            style="Panel.TLabel",
            font=("TkDefaultFont", 10, "bold"),
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 4))

        self._add_entry(control_panel, "X cm", self.x_var, 2)
        self._add_entry(control_panel, "Y cm", self.y_var, 3)
        self._add_entry(control_panel, "Z cm", self.z_var, 4)
        self._add_entry(control_panel, "Duration s", self.duration_var, 5)

        ttk.Label(
            control_panel,
            text="Titik drop",
            style="Panel.TLabel",
            font=("TkDefaultFont", 10, "bold"),
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(10, 4))

        self._add_entry(control_panel, "Drop X cm", self.drop_x_var, 7)
        self._add_entry(control_panel, "Drop Y cm", self.drop_y_var, 8)
        self._add_entry(control_panel, "Drop Z cm", self.drop_z_var, 9)
        self._add_entry(control_panel, "Drop Dur s", self.drop_duration_var, 10)

        ttk.Label(
            control_panel,
            text="Double-click Top View",
            style="Panel.TLabel",
        ).grid(row=11, column=0, columnspan=2, sticky="w", pady=(10, 2))

        mode_frame = ttk.Frame(control_panel, style="Panel.TFrame")
        mode_frame.grid(row=12, column=0, columnspan=2, sticky="ew")

        ttk.Radiobutton(
            mode_frame,
            text="Target",
            variable=self.point_mode_var,
            value="target",
        ).pack(side=tk.LEFT)

        ttk.Radiobutton(
            mode_frame,
            text="Drop",
            variable=self.point_mode_var,
            value="drop",
        ).pack(side=tk.LEFT, padx=(12, 0))

        send_btn = ttk.Button(
            control_panel,
            text="Kirim Pick + Drop",
            style="Primary.TButton",
            command=self.send_pick_drop,
        )
        send_btn.grid(row=13, column=0, columnspan=2, sticky="ew", pady=(12, 4))

        pick_only_btn = ttk.Button(
            control_panel,
            text="Kirim Pick Saja",
            command=self.send_xyz,
        )
        pick_only_btn.grid(row=14, column=0, columnspan=2, sticky="ew", pady=4)

        home_btn = ttk.Button(
            control_panel,
            text="Home",
            command=self.node.send_home,
        )
        home_btn.grid(row=15, column=0, sticky="ew", pady=4, padx=(0, 4))

        stop_btn = ttk.Button(
            control_panel,
            text="Stop",
            style="Danger.TButton",
            command=self.node.send_stop,
        )
        stop_btn.grid(row=15, column=1, sticky="ew", pady=4, padx=(4, 0))

        open_btn = ttk.Button(
            control_panel,
            text="Gripper Buka",
            command=lambda: self.node.send_gripper(30.0),
        )
        open_btn.grid(row=16, column=0, sticky="ew", pady=4, padx=(0, 4))

        close_btn = ttk.Button(
            control_panel,
            text="Gripper Tutup",
            command=lambda: self.node.send_gripper(130.0),
        )
        close_btn.grid(row=16, column=1, sticky="ew", pady=4, padx=(4, 0))

        ttk.Separator(control_panel).grid(
            row=17,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=10,
        )

        ttk.Label(
            control_panel,
            text="Posisi Aktual",
            style="Panel.TLabel",
            font=("TkDefaultFont", 11, "bold"),
        ).grid(row=18, column=0, columnspan=2, sticky="w")

        ttk.Label(
            control_panel,
            textvariable=self.pose_var,
            style="Panel.TLabel",
            justify=tk.LEFT,
        ).grid(row=19, column=0, columnspan=2, sticky="w", pady=(6, 10))

        ttk.Label(
            control_panel,
            text="Servo Aktual",
            style="Panel.TLabel",
            font=("TkDefaultFont", 11, "bold"),
        ).grid(row=20, column=0, columnspan=2, sticky="w")

        ttk.Label(
            control_panel,
            textvariable=self.servo_var,
            style="Panel.TLabel",
            justify=tk.LEFT,
        ).grid(row=21, column=0, columnspan=2, sticky="w", pady=(6, 10))

        ttk.Label(
            control_panel,
            text="Status",
            style="Panel.TLabel",
            font=("TkDefaultFont", 11, "bold"),
        ).grid(row=22, column=0, columnspan=2, sticky="w")

        self.status_label = ttk.Label(
            control_panel,
            textvariable=self.status_var,
            style="Panel.TLabel",
            justify=tk.LEFT,
            wraplength=260,
        )
        self.status_label.grid(row=23, column=0, columnspan=2, sticky="w", pady=(6, 0))

        views = ttk.Frame(main)
        views.grid(row=0, column=1, sticky="nsew")
        views.columnconfigure(0, weight=3)
        views.columnconfigure(1, weight=2)
        views.rowconfigure(1, weight=2)
        views.rowconfigure(3, weight=1)

        ttk.Label(
            views,
            text="Camera YOLO",
            style="Title.TLabel",
        ).grid(row=0, column=0, sticky="w", padx=(0, 6), pady=(0, 8))

        ttk.Label(
            views,
            textvariable=self.vision_status_var,
            style="Value.TLabel",
        ).grid(row=0, column=1, sticky="e", padx=(6, 0), pady=(0, 8))

        self.camera_canvas = tk.Canvas(
            views,
            background="#101828",
            highlightthickness=1,
            highlightbackground="#d0d5dd",
            width=560,
            height=300,
        )
        self.camera_canvas.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=(0, 6),
            pady=(0, 12),
        )
        self.camera_canvas.bind("<Button-1>", self._camera_canvas_click)

        vision_panel = ttk.Frame(views, style="Panel.TFrame", padding=10)
        vision_panel.grid(
            row=1,
            column=1,
            sticky="nsew",
            padx=(6, 0),
            pady=(0, 12),
        )
        vision_panel.columnconfigure(0, weight=1)
        vision_panel.rowconfigure(1, weight=1)

        ttk.Label(
            vision_panel,
            text="Objek terdeteksi",
            style="Panel.TLabel",
            font=("TkDefaultFont", 11, "bold"),
        ).grid(row=0, column=0, sticky="w")

        self.detection_tree = ttk.Treeview(
            vision_panel,
            columns=("label", "conf", "pixel", "robot"),
            show="headings",
            height=5,
        )
        self.detection_tree.heading("label", text="Label")
        self.detection_tree.heading("conf", text="Conf")
        self.detection_tree.heading("pixel", text="Pixel")
        self.detection_tree.heading("robot", text="Robot X,Y")
        self.detection_tree.column("label", width=90, anchor=tk.W)
        self.detection_tree.column("conf", width=52, anchor=tk.CENTER)
        self.detection_tree.column("pixel", width=92, anchor=tk.CENTER)
        self.detection_tree.column("robot", width=110, anchor=tk.CENTER)
        self.detection_tree.grid(row=1, column=0, sticky="nsew", pady=(6, 8))
        self.detection_tree.bind("<<TreeviewSelect>>", self._on_detection_selected)

        self.pick_detected_btn = ttk.Button(
            vision_panel,
            text="Pick Objek Terdeteksi",
            style="Primary.TButton",
            command=self.send_pick_detected,
            state=tk.DISABLED,
        )
        self.pick_detected_btn.grid(row=2, column=0, sticky="ew", pady=(0, 6))

        ttk.Label(
            vision_panel,
            textvariable=self.selected_object_var,
            style="Panel.TLabel",
            wraplength=330,
            justify=tk.LEFT,
        ).grid(row=3, column=0, sticky="w", pady=(0, 8))

        ttk.Separator(vision_panel).grid(row=4, column=0, sticky="ew", pady=6)

        calib_header = ttk.Frame(vision_panel, style="Panel.TFrame")
        calib_header.grid(row=5, column=0, sticky="ew")
        calib_header.columnconfigure(1, weight=1)

        ttk.Label(
            calib_header,
            text="Kalibrasi 4 titik",
            style="Panel.TLabel",
            font=("TkDefaultFont", 11, "bold"),
        ).grid(row=0, column=0, sticky="w")

        ttk.Checkbutton(
            calib_header,
            text="Mode klik",
            variable=self.calib_mode_var,
            command=self._toggle_calibration_mode,
        ).grid(row=0, column=1, sticky="e")

        calib_entries = ttk.Frame(vision_panel, style="Panel.TFrame")
        calib_entries.grid(row=6, column=0, sticky="ew", pady=(6, 4))
        for col in range(5):
            calib_entries.columnconfigure(col, weight=1 if col in [1, 3] else 0)

        ttk.Label(calib_entries, text="Titik", style="Panel.TLabel").grid(row=0, column=0)
        ttk.Label(calib_entries, text="X cm", style="Panel.TLabel").grid(row=0, column=1)
        ttk.Label(calib_entries, text="Y cm", style="Panel.TLabel").grid(row=0, column=3)

        for idx in range(4):
            row = idx + 1
            ttk.Label(
                calib_entries,
                text=f"P{idx + 1}",
                style="Panel.TLabel",
            ).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Entry(
                calib_entries,
                textvariable=self.calib_x_vars[idx],
                width=8,
            ).grid(row=row, column=1, sticky="ew", padx=(4, 8), pady=2)
            ttk.Label(calib_entries, text=",", style="Panel.TLabel").grid(row=row, column=2)
            ttk.Entry(
                calib_entries,
                textvariable=self.calib_y_vars[idx],
                width=8,
            ).grid(row=row, column=3, sticky="ew", padx=(8, 0), pady=2)

        calib_buttons = ttk.Frame(vision_panel, style="Panel.TFrame")
        calib_buttons.grid(row=7, column=0, sticky="ew", pady=(4, 4))
        calib_buttons.columnconfigure(0, weight=1)
        calib_buttons.columnconfigure(1, weight=1)

        ttk.Button(
            calib_buttons,
            text="Reset Titik Kamera",
            command=self._reset_calibration_points,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))

        ttk.Button(
            calib_buttons,
            text="Simpan Kalibrasi",
            command=self._save_calibration,
        ).grid(row=0, column=1, sticky="ew", padx=(4, 0))

        ttk.Label(
            vision_panel,
            textvariable=self.calibration_status_var,
            style="Panel.TLabel",
            wraplength=330,
            justify=tk.LEFT,
        ).grid(row=8, column=0, sticky="w")

        ttk.Label(
            views,
            text="Top View (X-Y)",
            style="Title.TLabel",
        ).grid(row=2, column=0, sticky="w", padx=(0, 6), pady=(0, 8))

        ttk.Label(
            views,
            text="Back View (X-Z)",
            style="Title.TLabel",
        ).grid(row=2, column=1, sticky="w", padx=(6, 0), pady=(0, 8))

        self.top_canvas = tk.Canvas(
            views,
            background="#ffffff",
            highlightthickness=1,
            highlightbackground="#d0d5dd",
        )
        self.top_canvas.grid(row=3, column=0, sticky="nsew", padx=(0, 6))
        self.top_canvas.bind("<Double-Button-1>", self.send_xyz_from_top_view)

        self.back_canvas = tk.Canvas(
            views,
            background="#ffffff",
            highlightthickness=1,
            highlightbackground="#d0d5dd",
        )
        self.back_canvas.grid(row=3, column=1, sticky="nsew", padx=(6, 0))

    def _add_entry(self, parent, label, variable, row):
        ttk.Label(parent, text=label, style="Panel.TLabel").grid(
            row=row,
            column=0,
            sticky="w",
            pady=5,
        )
        entry = ttk.Entry(parent, textvariable=variable, width=12)
        entry.grid(row=row, column=1, sticky="ew", pady=5, padx=(8, 0))
        entry.bind("<Return>", lambda _event: self.send_pick_drop())
        return entry

    def _read_target_inputs(self):
        x = float(self.x_var.get())
        y = float(self.y_var.get())
        z = float(self.z_var.get())
        duration = float(self.duration_var.get())
        if duration <= 0.0:
            raise ValueError("Duration target harus lebih besar dari 0")

        return x, y, z, duration

    def _read_drop_inputs(self):
        x = float(self.drop_x_var.get())
        y = float(self.drop_y_var.get())
        z = float(self.drop_z_var.get())
        duration = float(self.drop_duration_var.get())
        if duration <= 0.0:
            raise ValueError("Duration drop harus lebih besar dari 0")

        return x, y, z, duration

    def _sync_drop_marker_from_inputs(self, *_args):
        try:
            x = float(self.drop_x_var.get())
            y = float(self.drop_y_var.get())
            z = float(self.drop_z_var.get())
        except ValueError:
            return

        self.node.last_drop = {
            "x": x,
            "y": y,
            "z": z,
        }

    def send_xyz(self):
        try:
            x, y, z, duration = self._read_target_inputs()
        except ValueError as exc:
            self.node.last_error = f"Input tidak valid: {exc}"
            self.status_var.set(self.node.last_error)
            return

        self.node.last_error = ""
        self.node.send_xyz(x, y, z, duration)
        self._update_labels()

    def send_pick_drop(self):
        try:
            pick_x, pick_y, pick_z, pick_duration = self._read_target_inputs()
            drop_x, drop_y, drop_z, drop_duration = self._read_drop_inputs()
        except ValueError as exc:
            self.node.last_error = f"Input tidak valid: {exc}"
            self.status_var.set(self.node.last_error)
            return

        self.node.last_error = ""
        self.node.send_pick_drop(
            pick_x,
            pick_y,
            pick_z,
            pick_duration,
            drop_x,
            drop_y,
            drop_z,
            drop_duration,
        )
        self._update_labels()

    def send_xyz_from_top_view(self, event):
        world = self._top_view_canvas_to_world(event.x, event.y)
        if world is None:
            self.node.last_error = "Double click harus di dalam area grid Top View"
            self.status_var.set(self.node.last_error)
            return

        x, y = world

        if self.point_mode_var.get() == "drop":
            try:
                drop_z = float(self.drop_z_var.get())
            except ValueError as exc:
                self.node.last_error = f"Input drop tidak valid: {exc}"
                self.status_var.set(self.node.last_error)
                return

            self.drop_x_var.set(f"{x:.2f}")
            self.drop_y_var.set(f"{y:.2f}")
            self.node.last_error = ""
            self.node.last_drop = {
                "x": x,
                "y": y,
                "z": drop_z,
            }
            self.node.status_text = (
                f"Titik drop dari Top View: X={x:.2f}, Y={y:.2f}, "
                f"Z={drop_z:.2f}"
            )
            self._update_labels()
            return

        self.x_var.set(f"{x:.2f}")
        self.y_var.set(f"{y:.2f}")
        self.send_pick_drop()

        if not self.node.last_error:
            z = float(self.z_var.get())
            duration = float(self.duration_var.get())
            self.node.status_text = (
                f"Target dari Top View + Pick Drop: X={x:.2f}, Y={y:.2f}, "
                f"Z={z:.2f}, Duration={duration:.2f}s"
            )
            self._update_labels()

    def send_pick_detected(self):
        detection = self._get_selected_detection()
        if detection is None:
            self.node.last_error = "Tidak ada objek terdeteksi yang dipilih"
            self.status_var.set(self.node.last_error)
            return

        robot = detection.get("robot")
        if robot is None:
            self.node.last_error = "Objek belum punya koordinat robot. Simpan kalibrasi dulu."
            self.status_var.set(self.node.last_error)
            return

        try:
            pick_z = float(self.z_var.get())
            pick_duration = float(self.duration_var.get())
            if pick_duration <= 0.0:
                raise ValueError("Duration target harus lebih besar dari 0")
            drop_x, drop_y, drop_z, drop_duration = self._read_drop_inputs()
        except ValueError as exc:
            self.node.last_error = f"Input tidak valid: {exc}"
            self.status_var.set(self.node.last_error)
            return

        pick_x, pick_y = robot
        self.x_var.set(f"{pick_x:.2f}")
        self.y_var.set(f"{pick_y:.2f}")
        self.node.last_error = ""
        self.node.send_pick_drop(
            pick_x,
            pick_y,
            pick_z,
            pick_duration,
            drop_x,
            drop_y,
            drop_z,
            drop_duration,
        )
        self.node.status_text = (
            f"Pick objek {detection['label']} | "
            f"X={pick_x:.2f}, Y={pick_y:.2f}, Z={pick_z:.2f}"
        )
        self._update_labels()

    def _load_calibration(self):
        if not os.path.exists(CALIBRATION_FILE):
            self.calibration_status_var.set("Kalibrasi: belum ada")
            return

        try:
            with open(CALIBRATION_FILE, "r", encoding="utf-8") as file_obj:
                data = json.load(file_obj)

            image_points = self._normalize_points(data.get("image_points", []))
            robot_points = self._normalize_points(data.get("robot_points", []))
            if len(image_points) != 4 or len(robot_points) != 4:
                raise ValueError("file kalibrasi harus berisi 4 titik image dan 4 titik robot")

            self.calibration_image_points = image_points
            self.calibration_robot_points = robot_points
            for idx, point in enumerate(robot_points):
                self.calib_x_vars[idx].set(f"{point[0]:.2f}")
                self.calib_y_vars[idx].set(f"{point[1]:.2f}")

            matrix = data.get("homography")
            if matrix is not None:
                self.homography = np.array(matrix, dtype=np.float64)
                if self.homography.shape != (3, 3):
                    raise ValueError("matrix homography harus 3x3")
            else:
                self._compute_homography(robot_points)

            self.calibration_version += 1
            self.calibration_status_var.set(
                f"Kalibrasi: aktif dari {os.path.basename(CALIBRATION_FILE)}"
            )
        except Exception as exc:
            self.homography = None
            self.calibration_status_var.set(f"Kalibrasi: gagal load ({exc})")

    def _normalize_points(self, points):
        normalized = []
        for point in points:
            if len(point) < 2:
                continue
            normalized.append([float(point[0]), float(point[1])])
        return normalized

    def _read_calibration_robot_points(self):
        points = []
        for idx in range(4):
            x = float(self.calib_x_vars[idx].get())
            y = float(self.calib_y_vars[idx].get())
            points.append([x, y])
        return points

    def _compute_homography(self, robot_points):
        if len(self.calibration_image_points) != 4 or len(robot_points) != 4:
            raise ValueError("butuh tepat 4 titik kamera dan 4 titik robot")

        source = np.array(self.calibration_image_points, dtype=np.float32)
        target = np.array(robot_points, dtype=np.float32)
        matrix, _mask = cv2.findHomography(source, target, 0)
        if matrix is None:
            raise ValueError("cv2.findHomography gagal")

        self.homography = matrix
        self.calibration_robot_points = robot_points

    def _save_calibration(self):
        try:
            if len(self.calibration_image_points) != 4:
                raise ValueError("klik 4 titik pada kamera dulu")

            robot_points = self._read_calibration_robot_points()
            self._compute_homography(robot_points)

            data = {
                "image_points": self.calibration_image_points,
                "robot_points": robot_points,
                "homography": self.homography.tolist(),
            }
            with open(CALIBRATION_FILE, "w", encoding="utf-8") as file_obj:
                json.dump(data, file_obj, indent=2)

            self.node.last_error = ""
            self.calibration_version += 1
            self.calibration_status_var.set(
                f"Kalibrasi: tersimpan ({len(self.calibration_image_points)}/4 titik)"
            )
            self._refresh_detection_table(force=True)
        except Exception as exc:
            self.node.last_error = f"Kalibrasi gagal: {exc}"
            self.status_var.set(self.node.last_error)

    def _toggle_calibration_mode(self):
        if self.calib_mode_var.get():
            self.calibration_status_var.set(
                f"Kalibrasi: mode klik aktif ({len(self.calibration_image_points)}/4 titik)"
            )
        else:
            self.calibration_status_var.set(
                f"Kalibrasi: mode klik mati ({len(self.calibration_image_points)}/4 titik)"
            )

    def _reset_calibration_points(self):
        self.calibration_image_points = []
        self.calibration_robot_points = []
        self.homography = None
        self.calibration_version += 1
        self.calibration_status_var.set("Kalibrasi: titik kamera direset")
        self._refresh_detection_table(force=True)

    def _camera_canvas_click(self, event):
        if not self.calib_mode_var.get():
            return

        image_point = self._camera_canvas_to_image(event.x, event.y)
        if image_point is None:
            self.node.last_error = "Klik kalibrasi harus di dalam gambar kamera"
            self.status_var.set(self.node.last_error)
            return

        if len(self.calibration_image_points) >= 4:
            self.node.last_error = "Sudah ada 4 titik kamera. Reset jika ingin ulang."
            self.status_var.set(self.node.last_error)
            return

        self.calibration_image_points.append([image_point[0], image_point[1]])
        self.calibration_status_var.set(
            f"Kalibrasi: titik kamera {len(self.calibration_image_points)}/4"
        )
        self.calibration_version += 1
        self._refresh_detection_table(force=True)

    def _camera_canvas_to_image(self, canvas_x, canvas_y):
        if self.camera_display is None:
            return None

        offset_x = self.camera_display["offset_x"]
        offset_y = self.camera_display["offset_y"]
        display_width = self.camera_display["display_width"]
        display_height = self.camera_display["display_height"]
        scale = self.camera_display["scale"]

        if not (offset_x <= canvas_x <= offset_x + display_width):
            return None

        if not (offset_y <= canvas_y <= offset_y + display_height):
            return None

        image_x = (canvas_x - offset_x) / scale
        image_y = (canvas_y - offset_y) / scale
        return image_x, image_y

    def _pixel_to_robot(self, pixel_x, pixel_y):
        if self.homography is None:
            return None

        point = np.array([[[float(pixel_x), float(pixel_y)]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(point, self.homography)
        robot_x = float(mapped[0][0][0])
        robot_y = float(mapped[0][0][1])
        return robot_x, robot_y

    def _refresh_detection_table(self, force=False):
        if not hasattr(self, "detection_tree"):
            return

        needs_refresh = (
            force
            or self.last_detection_seq_drawn != self.node.detections_seq
            or self.last_calibration_version_drawn != self.calibration_version
        )

        if needs_refresh:
            previous_index = self.selected_detection_index
            self.visual_detections = self._build_visual_detections()

            for item_id in self.detection_tree.get_children():
                self.detection_tree.delete(item_id)

            for idx, detection in enumerate(self.visual_detections):
                center_x, center_y = detection["center"]
                robot = detection.get("robot")
                robot_text = (
                    "Belum kalibrasi"
                    if robot is None
                    else f"{robot[0]:.1f}, {robot[1]:.1f}"
                )
                self.detection_tree.insert(
                    "",
                    tk.END,
                    iid=str(idx),
                    values=(
                        detection["label"],
                        f"{detection['confidence']:.2f}",
                        f"{center_x:.0f}, {center_y:.0f}",
                        robot_text,
                    ),
                )

            if self.visual_detections:
                if previous_index is None or previous_index >= len(self.visual_detections):
                    previous_index = 0
                self.selected_detection_index = previous_index
                item_id = str(previous_index)
                self.detection_tree.selection_set(item_id)
                self.detection_tree.focus(item_id)
            else:
                self.selected_detection_index = None

            self.last_detection_seq_drawn = self.node.detections_seq
            self.last_calibration_version_drawn = self.calibration_version

        self._update_selected_object_label()
        self._update_pick_detected_button()

    def _build_visual_detections(self):
        detections = []
        for idx, detection in enumerate(self.node.detections):
            center_x, center_y = detection["center"]
            robot = None
            try:
                robot = self._pixel_to_robot(center_x, center_y)
            except Exception as exc:
                self.calibration_status_var.set(f"Kalibrasi: error transform ({exc})")

            item = dict(detection)
            item["display_index"] = idx
            item["robot"] = robot
            detections.append(item)
        return detections

    def _on_detection_selected(self, _event):
        selection = self.detection_tree.selection()
        if not selection:
            self.selected_detection_index = None
        else:
            self.selected_detection_index = int(selection[0])

        self._update_selected_object_label()
        self._update_pick_detected_button()

    def _get_selected_detection(self):
        if self.selected_detection_index is None:
            return None

        if not (0 <= self.selected_detection_index < len(self.visual_detections)):
            return None

        return self.visual_detections[self.selected_detection_index]

    def _update_selected_object_label(self):
        if not hasattr(self, "selected_object_var"):
            return

        detection = self._get_selected_detection()
        if detection is None:
            self.selected_object_var.set("Objek: belum ada")
            return

        center_x, center_y = detection["center"]
        robot = detection.get("robot")
        try:
            pick_z = float(self.z_var.get())
            z_text = f"{pick_z:.2f}"
        except ValueError:
            z_text = "invalid"

        if robot is None:
            coord_text = "Robot X-Y: belum kalibrasi"
        else:
            coord_text = f"Robot X={robot[0]:.2f}, Y={robot[1]:.2f}, Z={z_text}"

        self.selected_object_var.set(
            f"Objek: {detection['label']} ({detection['confidence']:.2f})\n"
            f"Pixel: {center_x:.0f}, {center_y:.0f}\n"
            f"{coord_text}"
        )

    def _update_pick_detected_button(self):
        if not hasattr(self, "pick_detected_btn"):
            return

        detection = self._get_selected_detection()
        enabled = detection is not None and detection.get("robot") is not None

        if enabled:
            try:
                _pick_z = float(self.z_var.get())
                pick_duration = float(self.duration_var.get())
                if pick_duration <= 0.0:
                    raise ValueError
                self._read_drop_inputs()
            except ValueError:
                enabled = False

        self.pick_detected_btn.configure(state=tk.NORMAL if enabled else tk.DISABLED)

    def _spin_ros(self):
        if rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.0)
            self._refresh_detection_table()
            self._update_labels()

        self.root.after(20, self._spin_ros)

    def _update_labels(self):
        pose = self.node.current_xyz
        self.pose_var.set(
            f"X      = {pose['x']:.2f} cm\n"
            f"Y      = {pose['y']:.2f} cm\n"
            f"Z      = {pose['z']:.2f} cm\n"
            f"Pitch  = {pose['pitch']:.2f} deg"
        )

        q = self.node.model_angles
        self.servo_var.set(
            f"Base     = {q['base']:.2f} deg\n"
            f"Shoulder = {q['shoulder']:.2f} deg\n"
            f"Elbow    = {q['elbow']:.2f} deg\n"
            f"Wrist    = {q['wrist']:.2f} deg\n"
            f"Gripper  = {self.node.gripper_angle:.2f} deg"
        )

        status = self.node.last_error or self.node.status_text
        self.status_var.set(status)
        self.vision_status_var.set(f"Vision: {self.node.vision_status_text}")

    def _draw_visuals(self):
        self._draw_camera_view()
        positions = self.kin.joint_positions(self.node.model_angles)
        self._draw_top_view(self.top_canvas, positions)
        self._draw_back_view(self.back_canvas, positions)
        self.root.after(50, self._draw_visuals)

    def _draw_camera_view(self):
        canvas = self.camera_canvas
        width = max(canvas.winfo_width(), 320)
        height = max(canvas.winfo_height(), 240)
        canvas.delete("all")

        frame = self.node.camera_frame
        if frame is None:
            self.camera_display = None
            canvas.create_text(
                width / 2,
                height / 2,
                text="Menunggu frame kamera...",
                fill="#ffffff",
                font=("TkDefaultFont", 11, "bold"),
            )
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_height, image_width = frame_rgb.shape[:2]
        scale = min(width / image_width, height / image_height)
        display_width = max(1, int(image_width * scale))
        display_height = max(1, int(image_height * scale))
        offset_x = (width - display_width) / 2.0
        offset_y = (height - display_height) / 2.0

        image = PilImage.fromarray(frame_rgb)
        image = image.resize((display_width, display_height), IMAGE_RESAMPLE)
        self.camera_photo = ImageTk.PhotoImage(image=image)
        canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=self.camera_photo)
        canvas.create_rectangle(
            offset_x,
            offset_y,
            offset_x + display_width,
            offset_y + display_height,
            outline="#d0d5dd",
        )

        self.camera_display = {
            "offset_x": offset_x,
            "offset_y": offset_y,
            "display_width": display_width,
            "display_height": display_height,
            "scale": scale,
            "image_width": image_width,
            "image_height": image_height,
        }

        for idx, point in enumerate(self.calibration_image_points):
            point_x = offset_x + point[0] * scale
            point_y = offset_y + point[1] * scale
            canvas.create_oval(
                point_x - 6,
                point_y - 6,
                point_x + 6,
                point_y + 6,
                fill="#12b76a",
                outline="#ffffff",
                width=2,
            )
            canvas.create_text(
                point_x + 14,
                point_y - 10,
                text=f"P{idx + 1}",
                fill="#ffffff",
                font=("TkDefaultFont", 9, "bold"),
            )

        for idx, detection in enumerate(self.visual_detections):
            center_x, center_y = detection["center"]
            point_x = offset_x + center_x * scale
            point_y = offset_y + center_y * scale
            selected = idx == self.selected_detection_index
            color = "#f04438" if selected else "#fdb022"
            radius = 7 if selected else 5
            canvas.create_oval(
                point_x - radius,
                point_y - radius,
                point_x + radius,
                point_y + radius,
                outline=color,
                width=3 if selected else 2,
            )

    def _draw_grid(self, canvas, title, x_label, y_label):
        width = max(canvas.winfo_width(), 320)
        height = max(canvas.winfo_height(), 320)
        canvas.delete("all")

        margin = 34
        canvas.create_rectangle(
            margin,
            margin,
            width - margin,
            height - margin,
            outline="#d0d5dd",
        )

        for i in range(1, 4):
            x = margin + (width - margin * 2) * i / 4
            y = margin + (height - margin * 2) * i / 4
            canvas.create_line(x, margin, x, height - margin, fill="#eef2f6")
            canvas.create_line(margin, y, width - margin, y, fill="#eef2f6")

        canvas.create_text(12, height / 2, text=y_label, angle=90, fill="#475467")
        canvas.create_text(width / 2, height - 12, text=x_label, fill="#475467")
        canvas.create_text(width / 2, 16, text=title, fill="#17202a", font=("TkDefaultFont", 11, "bold"))

        return width, height, margin

    def _top_view_geometry(self, canvas):
        width = max(canvas.winfo_width(), 320)
        height = max(canvas.winfo_height(), 320)
        margin = 34
        scale = min((width - margin * 2) / 80.0, (height - margin * 2) / 80.0)
        origin_x = width / 2.0
        origin_y = height - margin - 8.0
        return width, height, margin, scale, origin_x, origin_y

    def _top_view_canvas_to_world(self, canvas_x, canvas_y):
        width, height, margin, scale, origin_x, origin_y = self._top_view_geometry(
            self.top_canvas
        )

        if not (margin <= canvas_x <= width - margin):
            return None

        if not (margin <= canvas_y <= height - margin):
            return None

        x = (canvas_x - origin_x) / scale
        y = (origin_y - canvas_y) / scale
        return x, y

    def _draw_top_view(self, canvas, positions):
        width, height, margin = self._draw_grid(canvas, "Top View", "X cm", "Y cm")
        _width, _height, _margin, scale, origin_x, origin_y = self._top_view_geometry(
            canvas
        )

        def point(pos):
            return (
                origin_x + pos["x"] * scale,
                origin_y - pos["y"] * scale,
            )

        self._draw_axes(canvas, origin_x, origin_y, scale, top_view=True)
        self._draw_robot(canvas, positions, point)

        tip = self.node.current_xyz
        tip_x = origin_x + tip["x"] * scale
        tip_y = origin_y - tip["y"] * scale
        self._draw_marker(canvas, tip_x, tip_y, "#1570ef", "actual")

        if self.node.last_target:
            target = self.node.last_target
            target_x = origin_x + target["x"] * scale
            target_y = origin_y - target["y"] * scale
            self._draw_cross(canvas, target_x, target_y, "#dc6803", "target")

        if self.node.last_drop:
            drop = self.node.last_drop
            drop_x = origin_x + drop["x"] * scale
            drop_y = origin_y - drop["y"] * scale
            self._draw_cross(canvas, drop_x, drop_y, "#039855", "drop")

        for idx, detection in enumerate(self.visual_detections):
            robot = detection.get("robot")
            if robot is None:
                continue
            obj_x = origin_x + robot[0] * scale
            obj_y = origin_y - robot[1] * scale
            self._draw_object_marker(
                canvas,
                obj_x,
                obj_y,
                idx == self.selected_detection_index,
                f"obj{idx + 1}",
            )

    def _draw_back_view(self, canvas, positions):
        width, height, margin = self._draw_grid(canvas, "Back View", "X cm", "Z cm")
        scale = min((width - margin * 2) / 80.0, (height - margin * 2) / 50.0)
        origin_x = width / 2.0
        origin_y = height - margin - 8.0

        def point(pos):
            return (
                origin_x + pos["x"] * scale,
                origin_y - pos["z"] * scale,
            )

        self._draw_axes(canvas, origin_x, origin_y, scale, top_view=False)
        self._draw_robot(canvas, positions, point)

        tip = self.node.current_xyz
        tip_x = origin_x + tip["x"] * scale
        tip_y = origin_y - tip["z"] * scale
        self._draw_marker(canvas, tip_x, tip_y, "#1570ef", "actual")

        if self.node.last_target:
            target = self.node.last_target
            target_x = origin_x + target["x"] * scale
            target_y = origin_y - target["z"] * scale
            self._draw_cross(canvas, target_x, target_y, "#dc6803", "target")

        if self.node.last_drop:
            drop = self.node.last_drop
            drop_x = origin_x + drop["x"] * scale
            drop_y = origin_y - drop["z"] * scale
            self._draw_cross(canvas, drop_x, drop_y, "#039855", "drop")

        pick_z = self._current_pick_z()
        if pick_z is not None:
            for idx, detection in enumerate(self.visual_detections):
                robot = detection.get("robot")
                if robot is None:
                    continue
                obj_x = origin_x + robot[0] * scale
                obj_y = origin_y - pick_z * scale
                self._draw_object_marker(
                    canvas,
                    obj_x,
                    obj_y,
                    idx == self.selected_detection_index,
                    f"obj{idx + 1}",
                )

    def _draw_axes(self, canvas, origin_x, origin_y, scale, top_view):
        axis_color = "#98a2b3"
        canvas.create_line(origin_x - 35 * scale, origin_y, origin_x + 35 * scale, origin_y, fill=axis_color)
        canvas.create_line(origin_x, origin_y, origin_x, origin_y - 35 * scale, fill=axis_color)
        canvas.create_text(origin_x + 35 * scale + 10, origin_y, text="+X", fill="#667085")
        canvas.create_text(
            origin_x,
            origin_y - 35 * scale - 10,
            text="+Y" if top_view else "+Z",
            fill="#667085",
        )

    def _draw_robot(self, canvas, positions, point_fn):
        points = [point_fn(pos) for pos in positions]

        for idx in range(len(points) - 1):
            x1, y1 = points[idx]
            x2, y2 = points[idx + 1]
            canvas.create_line(
                x1,
                y1,
                x2,
                y2,
                width=5 if idx > 0 else 3,
                fill="#344054",
                capstyle=tk.ROUND,
            )

        colors = {
            "base": "#101828",
            "shoulder": "#475467",
            "elbow": "#12b76a",
            "wrist": "#7a5af8",
            "tip": "#f04438",
        }

        for pos, (x, y) in zip(positions, points):
            radius = 6 if pos["name"] != "tip" else 7
            canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill=colors.get(pos["name"], "#344054"),
                outline="#ffffff",
                width=2,
            )
            canvas.create_text(
                x,
                y - 13,
                text=pos["name"],
                fill="#344054",
                font=("TkDefaultFont", 8),
            )

    def _draw_marker(self, canvas, x, y, color, label):
        radius = 5
        canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            outline=color,
            width=2,
        )
        canvas.create_text(x + 22, y - 10, text=label, fill=color, font=("TkDefaultFont", 8))

    def _draw_cross(self, canvas, x, y, color, label):
        size = 7
        canvas.create_line(x - size, y - size, x + size, y + size, fill=color, width=2)
        canvas.create_line(x - size, y + size, x + size, y - size, fill=color, width=2)
        canvas.create_text(x + 22, y + 10, text=label, fill=color, font=("TkDefaultFont", 8))

    def _draw_object_marker(self, canvas, x, y, selected, label):
        color = "#f04438" if selected else "#fdb022"
        radius = 6 if selected else 5
        canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill=color,
            outline="#ffffff",
            width=2,
        )
        canvas.create_text(
            x + 24,
            y - 10,
            text=label,
            fill=color,
            font=("TkDefaultFont", 8, "bold" if selected else "normal"),
        )

    def _current_pick_z(self):
        try:
            return float(self.z_var.get())
        except ValueError:
            return None

    def close(self):
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main(args=None):
    rclpy.init(args=args)
    node = ArmXYZGuiNode()

    try:
        app = ArmXYZGuiApp(node)
        app.run()

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
