#!/usr/bin/env python3

import math
import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Tuple

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64, Float64MultiArray, Empty, String

import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo


# ============================================================
# ARM DRIVER NODE
# Robot Arm 4 DOF + 1 Gripper
#
# Sistem koordinat:
# X = kiri / kanan robot
# Y = depan robot
# Z = atas robot
#
# HOME fisik:
# Semua servo = 90 derajat
#
# HOME model:
# base     = 0 derajat
# shoulder = 90 derajat
# elbow    = -90 derajat
# wrist    = 0 derajat
#
# HOME end-effector:
# X = 0 cm
# Y = 24 cm
# Z = 22 cm
#
# Panjang link:
# L1 = 10 cm
# L2 = 12 cm
# L3 = 8 cm
# L4 = 16 cm
#
# Mode IK terbaru:
# User hanya input X Y Z.
# Saat menuju target, driver mencoba membuat gripper menghadap bawah.
# Setelah auto gripper close, wrist dikembalikan ke 0 derajat agar
# gripper lurus sejajar link wrist.
#
# PCA9685:
# CH0 = Base
# CH1 = Shoulder
# CH2 = Elbow
# CH3 = Wrist
# CH4 = Gripper
#
# Subscribe:
# /arm/target_xyz  Float64MultiArray [x, y, z, duration]
#                  Setelah target XYZ tercapai, gripper otomatis menutup.
# /arm/pick_drop_xyz Float64MultiArray
#                    [pick_x, pick_y, pick_z, pick_duration,
#                     drop_x, drop_y, drop_z, drop_duration]
# /arm/gripper     Float64
# /arm/home        Empty
# /arm/stop        Empty
#
# Publish:
# /arm/current_xyz Float64MultiArray [x, y, z, pitch]
# /arm/servo_angle Float64MultiArray [base, shoulder, elbow, wrist, gripper]
# /arm/status      String
# ============================================================


def clamp(value, low, high):
    return max(low, min(high, value))


def smoothstep(t):
    t = clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass
class ArmGeometry:
    L1: float = 8.0
    L2: float = 12.0
    L3: float = 8.0
    L4: float = 15.0


@dataclass
class JointCalibration:
    name: str
    channel: int

    model_home_deg: float
    servo_home_deg: float = 90.0

    direction: int = 1
    trim_deg: float = 0.0

    servo_min_deg: float = 0.0
    servo_max_deg: float = 180.0

    min_pulse: int = 600
    max_pulse: int = 2400

    def model_to_servo(self, model_deg: float) -> float:
        servo_deg = (
            self.servo_home_deg
            + self.direction * (model_deg - self.model_home_deg)
            + self.trim_deg
        )
        return clamp(servo_deg, self.servo_min_deg, self.servo_max_deg)

    def servo_to_model(self, servo_deg: float) -> float:
        return (
            self.model_home_deg
            + self.direction * (servo_deg - self.servo_home_deg - self.trim_deg)
        )

    def model_limits(self) -> Tuple[float, float]:
        a = self.servo_to_model(self.servo_min_deg)
        b = self.servo_to_model(self.servo_max_deg)
        return min(a, b), max(a, b)


class RobotKinematics:
    def __init__(self):
        self.geo = ArmGeometry()

    def forward(self, q: Dict[str, float]) -> Dict[str, float]:
        L1 = self.geo.L1
        L2 = self.geo.L2
        L3 = self.geo.L3
        L4 = self.geo.L4

        q0 = math.radians(q["base"])
        q1 = math.radians(q["shoulder"])
        q2 = math.radians(q["elbow"])
        q3 = math.radians(q["wrist"])

        r = (
            L2 * math.cos(q1)
            + L3 * math.cos(q1 + q2)
            + L4 * math.cos(q1 + q2 + q3)
        )

        z = (
            L1
            + L2 * math.sin(q1)
            + L3 * math.sin(q1 + q2)
            + L4 * math.sin(q1 + q2 + q3)
        )

        x = r * math.sin(q0)
        y = r * math.cos(q0)

        pitch = math.degrees(q1 + q2 + q3)

        return {
            "x": x,
            "y": y,
            "z": z,
            "pitch": pitch,
        }

    def inverse_xyz_fixed_pitch_candidates(
        self,
        x,
        y,
        z,
        pitch_deg,
    ) -> List[Dict[str, float]]:
        """
        IK dari X Y Z dengan pitch ujung gripper ditentukan.

        Dipakai agar gripper menghadap bawah saat menuju target.
        L4 dianggap sebagai link dari wrist ke ujung gripper, sehingga
        shoulder dan elbow menyelesaikan posisi wrist center.
        """

        L1 = self.geo.L1
        L2 = self.geo.L2
        L3 = self.geo.L3
        L4 = self.geo.L4

        r_tip = math.hypot(x, y)
        base = math.atan2(x, y)
        pitch = math.radians(pitch_deg)

        r_wrist = r_tip - L4 * math.cos(pitch)
        z_wrist = z - L4 * math.sin(pitch)
        z_rel = z_wrist - L1

        if r_wrist < 0.0:
            raise ValueError(
                f"Target terlalu dekat untuk pitch {pitch_deg:.1f} deg. "
                f"R wrist = {r_wrist:.2f} cm."
            )

        d2 = r_wrist * r_wrist + z_rel * z_rel
        d = math.sqrt(d2)

        reach_max = L2 + L3
        reach_min = abs(L3 - L2)

        if d > reach_max:
            raise ValueError(
                f"Target terlalu jauh untuk pitch {pitch_deg:.1f} deg. "
                f"Jarak shoulder ke wrist = {d:.2f} cm, "
                f"maksimum = {reach_max:.2f} cm."
            )

        if d < reach_min:
            raise ValueError(
                f"Target terlalu dekat untuk pitch {pitch_deg:.1f} deg. "
                f"Jarak shoulder ke wrist = {d:.2f} cm, "
                f"minimum = {reach_min:.2f} cm."
            )

        cos_elbow = (d2 - L2 * L2 - L3 * L3) / (2.0 * L2 * L3)
        cos_elbow = clamp(cos_elbow, -1.0, 1.0)

        solutions = []

        for sign, branch in [(-1.0, "elbow_negative"), (1.0, "elbow_positive")]:
            elbow = sign * math.acos(cos_elbow)

            shoulder = math.atan2(z_rel, r_wrist) - math.atan2(
                L3 * math.sin(elbow),
                L2 + L3 * math.cos(elbow),
            )

            wrist = pitch - shoulder - elbow

            solutions.append(
                {
                    "base": math.degrees(base),
                    "shoulder": math.degrees(shoulder),
                    "elbow": math.degrees(elbow),
                    "wrist": math.degrees(wrist),
                    "pitch_auto": pitch_deg,
                    "branch": f"gripper_down_{pitch_deg:.0f}_{branch}",
                }
            )

        return solutions

    def inverse_xyz_direct_candidates(self, x, y, z) -> List[Dict[str, float]]:
        """
        IK langsung dari X Y Z.

        User tidak input pitch.
        Wrist model dibuat 0 derajat.
        L3 + L4 dianggap sebagai satu link lurus dari elbow ke ujung gripper.

        Hasil:
        - base
        - shoulder
        - elbow
        - wrist = 0
        - pitch otomatis = shoulder + elbow + wrist
        """

        L1 = self.geo.L1
        L2 = self.geo.L2
        L3 = self.geo.L3
        L4 = self.geo.L4

        L34 = L3 + L4

        r = math.hypot(x, y)
        z_rel = z - L1

        base = math.atan2(x, y)

        d2 = r * r + z_rel * z_rel
        d = math.sqrt(d2)

        reach_max = L2 + L34
        reach_min = abs(L34 - L2)

        if d > reach_max:
            raise ValueError(
                f"Target terlalu jauh. Jarak shoulder ke tip = {d:.2f} cm, "
                f"maksimum = {reach_max:.2f} cm."
            )

        if d < reach_min:
            raise ValueError(
                f"Target terlalu dekat. Jarak shoulder ke tip = {d:.2f} cm, "
                f"minimum = {reach_min:.2f} cm."
            )

        cos_elbow = (d2 - L2 * L2 - L34 * L34) / (2.0 * L2 * L34)
        cos_elbow = clamp(cos_elbow, -1.0, 1.0)

        solutions = []

        for sign, branch in [(-1.0, "elbow_negative"), (1.0, "elbow_positive")]:
            elbow = sign * math.acos(cos_elbow)

            shoulder = math.atan2(z_rel, r) - math.atan2(
                L34 * math.sin(elbow),
                L2 + L34 * math.cos(elbow),
            )

            wrist = 0.0
            pitch_auto = shoulder + elbow + wrist

            solutions.append(
                {
                    "base": math.degrees(base),
                    "shoulder": math.degrees(shoulder),
                    "elbow": math.degrees(elbow),
                    "wrist": math.degrees(wrist),
                    "pitch_auto": math.degrees(pitch_auto),
                    "branch": branch,
                }
            )

        return solutions


class RobotArm:
    def __init__(self):
        # ====================================================
        # PCA9685 HARDWARE INIT
        # ====================================================
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(self.i2c)
        self.pca.frequency = 50

        self.kin = RobotKinematics()

        # ====================================================
        # KALIBRASI SESUAI ROBOT ANDA
        # ====================================================
        self.joints = {
            "base": JointCalibration(
                name="base",
                channel=0,
                model_home_deg=0.0,
                servo_home_deg=80.0,
                direction=-1,
                trim_deg=0.0,
                servo_min_deg=0.0,
                servo_max_deg=180.0,
                min_pulse=600,
                max_pulse=2400,
            ),

            "shoulder": JointCalibration(
                name="shoulder",
                channel=1,
                model_home_deg=90.0,
                servo_home_deg=90.0,
                direction=+1,
                trim_deg=0.0,
                servo_min_deg=15.0,
                servo_max_deg=185.0,
                min_pulse=600,
                max_pulse=2400,
            ),

            "elbow": JointCalibration(
                name="elbow",
                channel=2,
                model_home_deg=-90.0,
                servo_home_deg=90.0,
                direction=-1,
                trim_deg=0.0,
                servo_min_deg=15.0,
                servo_max_deg=170.0,
                min_pulse=600,
                max_pulse=2400,
            ),

            "wrist": JointCalibration(
                name="wrist",
                channel=3,
                model_home_deg=0.0,
                servo_home_deg=90.0,
                direction=-1,
                trim_deg=0.0,
                servo_min_deg=0.0,
                servo_max_deg=180.0,
                min_pulse=600,
                max_pulse=2400,
            ),
        }

        self.gripper = JointCalibration(
            name="gripper",
            channel=4,
            model_home_deg=90.0,
            servo_home_deg=90.0,
            direction=+1,
            trim_deg=0.0,
            servo_min_deg=30.0,
            servo_max_deg=150.0,
            min_pulse=600,
            max_pulse=2400,
        )

        self.hw_servos = {}

        for name, joint in self.joints.items():
            self.hw_servos[name] = servo.Servo(
                self.pca.channels[joint.channel],
                min_pulse=joint.min_pulse,
                max_pulse=joint.max_pulse,
            )

        self.hw_gripper = servo.Servo(
            self.pca.channels[self.gripper.channel],
            min_pulse=self.gripper.min_pulse,
            max_pulse=self.gripper.max_pulse,
        )

        self.q_home = {
            "base": 0.0,
            "shoulder": 90.0,
            "elbow": -90.0,
            "wrist": -90.0,
        }

        self.q_current = dict(self.q_home)
        self.gripper_current = 30.0
        self.gripper_open_duration = 0.3
        self.gripper_close_duration = 1.5
        self.gripper_rate_hz = 50
        self.stop_request = False
        self.last_pitch_auto = 0.0

    def print_model_limits(self):
        print()
        print("BATAS MODEL JOINT:")
        for name, joint in self.joints.items():
            lo, hi = joint.model_limits()
            print(f"{name:8s}: {lo:8.2f} sampai {hi:8.2f} deg")
        print()

    def write_servo_angle(self, name: str, servo_angle: float):
        servo_angle = clamp(servo_angle, 0.0, 180.0)
        self.hw_servos[name].angle = servo_angle

    def write_model_angles(self, q: Dict[str, float]):
        for name, model_angle in q.items():
            joint = self.joints[name]
            servo_angle = joint.model_to_servo(model_angle)
            self.write_servo_angle(name, servo_angle)

    def write_home_direct(self):
        for name in self.joints.keys():
            self.hw_servos[name].angle = 90.0

        self.hw_gripper.angle = 30.0

        self.q_current = dict(self.q_home)
        self.gripper_current = 30.0
        self.last_pitch_auto = 0.0

    def within_limit(self, q: Dict[str, float]) -> bool:
        for name, model_angle in q.items():
            lo, hi = self.joints[name].model_limits()

            if not (lo <= model_angle <= hi):
                print(
                    f"[LIMIT] {name}: {model_angle:.2f} deg, "
                    f"batas {lo:.2f} sampai {hi:.2f}"
                )
                return False

        return True

    def choose_solution(self, candidates, verbose=True):
        valid = []

        for q in candidates:
            q_test = {
                "base": q["base"],
                "shoulder": q["shoulder"],
                "elbow": q["elbow"],
                "wrist": q["wrist"],
            }

            if self.within_limit(q_test):
                valid.append(q)

        if not valid:
            if verbose:
                print()
                print("KANDIDAT IK:")
                for q in candidates:
                    print(q)

            raise ValueError("Tidak ada solusi IK yang masuk limit servo.")

        # Prioritaskan elbow negatif karena sesuai bentuk home robot.
        elbow_negative = [q for q in valid if q["elbow"] < 0.0]

        if elbow_negative:
            valid = elbow_negative

        def cost(q):
            return (
                abs(q["base"] - self.q_current["base"])
                + abs(q["shoulder"] - self.q_current["shoulder"])
                + abs(q["elbow"] - self.q_current["elbow"])
                + abs(q["wrist"] - self.q_current["wrist"])
            )

        return min(valid, key=cost)

    def move_joints(self, q_target: Dict[str, float], duration=2.0, rate_hz=50):
        if not self.within_limit(q_target):
            raise ValueError(f"Target joint keluar limit: {q_target}")

        self.stop_request = False

        q_start = dict(self.q_current)
        steps = max(1, int(duration * rate_hz))

        for i in range(1, steps + 1):
            if self.stop_request:
                break

            u = smoothstep(i / steps)

            q_now = {
                name: q_start[name] + (q_target[name] - q_start[name]) * u
                for name in self.joints.keys()
            }

            self.write_model_angles(q_now)
            self.q_current = dict(q_now)

            time.sleep(duration / steps)

        if not self.stop_request:
            self.q_current = dict(q_target)

    def move_xyz(self, x, y, z, duration=2.0):
        """
        Gerak berdasarkan input X Y Z saja.

        Tidak ada input pitch.

        Metode:
        - Driver mencoba pitch gripper menghadap bawah.
        - Pitch dicoba dari paling bawah ke lebih miring agar target
          tetap bisa dicapai oleh limit mekanik.
        """

        selected = None
        errors = []

        for pitch_deg in [-90.0, -80.0, -70.0, -60.0]:
            try:
                candidates = self.kin.inverse_xyz_fixed_pitch_candidates(
                    x,
                    y,
                    z,
                    pitch_deg,
                )
                selected = self.choose_solution(candidates, verbose=False)
                break
            except Exception as e:
                errors.append(f"{pitch_deg:.0f} deg: {e}")

        if selected is None:
            detail = " | ".join(errors)
            raise ValueError(
                "Tidak ada solusi IK dengan gripper menghadap bawah. "
                f"Detail: {detail}"
            )

        q_target = {
            "base": selected["base"],
            "shoulder": selected["shoulder"],
            "elbow": selected["elbow"],
            "wrist": selected["wrist"],
        }

        self.last_pitch_auto = selected["pitch_auto"]

        self.move_joints(q_target, duration)

        return q_target, self.last_pitch_auto, selected["branch"]

    def move_xyz_wrist_straight(self, x, y, z, duration=2.0):
        """
        Gerak ke X Y Z dengan wrist model = 0 derajat.

        Dipakai untuk membawa barang ke titik drop setelah gripper
        tertutup dan wrist sudah diluruskan.
        """

        candidates = self.kin.inverse_xyz_direct_candidates(x, y, z)
        selected = self.choose_solution(candidates)

        q_target = {
            "base": selected["base"],
            "shoulder": selected["shoulder"],
            "elbow": selected["elbow"],
            "wrist": selected["wrist"],
        }

        self.last_pitch_auto = selected["pitch_auto"]

        self.move_joints(q_target, duration)

        return q_target, self.last_pitch_auto, selected["branch"]

    def straighten_wrist(self, duration=0.7):
        """
        Kembalikan wrist ke 0 derajat agar L3 dan L4 lurus/sejajar.
        """

        q_target = dict(self.q_current)
        q_target["wrist"] = 0.0

        self.move_joints(q_target, duration)

        pose = self.current_pose()
        self.last_pitch_auto = pose["pitch"]

        return q_target

    def home(self):
        self.move_joints(self.q_home, duration=1.5)
        self.set_gripper(30.0)
        self.last_pitch_auto = 0.0

    def set_gripper(self, angle: float):
        angle = clamp(angle, self.gripper.servo_min_deg, self.gripper.servo_max_deg)
        self.hw_gripper.angle = angle
        self.gripper_current = angle

    def move_gripper(self, angle: float, duration=None, rate_hz=None):
        angle = clamp(angle, self.gripper.servo_min_deg, self.gripper.servo_max_deg)
        duration = self.gripper_close_duration if duration is None else float(duration)
        rate_hz = self.gripper_rate_hz if rate_hz is None else float(rate_hz)

        duration = max(0.0, duration)
        rate_hz = max(1.0, rate_hz)

        if duration <= 0.0:
            self.set_gripper(angle)
            return angle

        start_angle = float(self.gripper_current)
        steps = max(1, int(duration * rate_hz))

        for i in range(1, steps + 1):
            if self.stop_request:
                break

            u = smoothstep(i / steps)
            now = start_angle + (angle - start_angle) * u
            self.set_gripper(now)
            time.sleep(duration / steps)

        if not self.stop_request:
            self.set_gripper(angle)

        return self.gripper_current

    def gripper_open(self):
        self.move_gripper(30.0, self.gripper_open_duration)

    def gripper_close(self):
        self.move_gripper(120.0, self.gripper_close_duration)

    def current_pose(self):
        return self.kin.forward(self.q_current)

    def current_servo_angles(self):
        data = {}

        for name in self.joints:
            joint = self.joints[name]
            data[name] = joint.model_to_servo(self.q_current[name])

        data["gripper"] = self.gripper_current
        return data

    def stop(self):
        self.stop_request = True

    def release_all(self):
        for name in self.hw_servos:
            self.hw_servos[name].angle = None

        self.hw_gripper.angle = None

        try:
            self.pca.deinit()
        except Exception:
            pass


class ArmDriverNode(Node):
    def __init__(self):
        super().__init__("arm_driver_node")

        self.arm = RobotArm()
        self.lock = threading.Lock()

        self.pose_sub = self.create_subscription(
            Float64MultiArray,
            "/arm/target_xyz",
            self.target_xyz_callback,
            10,
        )

        self.pick_drop_sub = self.create_subscription(
            Float64MultiArray,
            "/arm/pick_drop_xyz",
            self.pick_drop_xyz_callback,
            10,
        )

        self.gripper_sub = self.create_subscription(
            Float64,
            "/arm/gripper",
            self.gripper_callback,
            10,
        )

        self.home_sub = self.create_subscription(
            Empty,
            "/arm/home",
            self.home_callback,
            10,
        )

        self.stop_sub = self.create_subscription(
            Empty,
            "/arm/stop",
            self.stop_callback,
            10,
        )

        self.pose_pub = self.create_publisher(
            Float64MultiArray,
            "/arm/current_xyz",
            10,
        )

        self.servo_pub = self.create_publisher(
            Float64MultiArray,
            "/arm/servo_angle",
            10,
        )

        self.status_pub = self.create_publisher(
            String,
            "/arm/status",
            10,
        )

        self.timer = self.create_timer(0.2, self.publish_state)

        self.get_logger().info("ARM DRIVER NODE AKTIF")
        self.get_logger().info("PCA9685: 50 Hz, pulse 600-2400 us")
        self.get_logger().info("Mode input: X Y Z saja")
        self.get_logger().info(
            "Mode IK: gripper menghadap bawah saat menuju target, "
            "lalu wrist lurus setelah gripping"
        )
        self.get_logger().info("Mengirim servo ke HOME 90 derajat...")

        try:
            self.arm.write_home_direct()
            self.arm.print_model_limits()

            pose = self.arm.current_pose()

            self.publish_status(
                f"HOME DIRECT OK | "
                f"X={pose['x']:.2f}, Y={pose['y']:.2f}, "
                f"Z={pose['z']:.2f}, Pitch={pose['pitch']:.2f}"
            )

        except Exception as e:
            self.publish_status(f"ERROR INIT: {e}")

    def publish_status(self, text):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(text)

    def publish_state(self):
        try:
            pose = self.arm.current_pose()
            servo_angles = self.arm.current_servo_angles()

            pose_msg = Float64MultiArray()
            pose_msg.data = [
                float(pose["x"]),
                float(pose["y"]),
                float(pose["z"]),
                float(pose["pitch"]),
            ]
            self.pose_pub.publish(pose_msg)

            servo_msg = Float64MultiArray()
            servo_msg.data = [
                float(servo_angles["base"]),
                float(servo_angles["shoulder"]),
                float(servo_angles["elbow"]),
                float(servo_angles["wrist"]),
                float(servo_angles["gripper"]),
            ]
            self.servo_pub.publish(servo_msg)

        except Exception as e:
            self.get_logger().error(f"Publish state error: {e}")

    def target_xyz_callback(self, msg):
        data = list(msg.data)

        if len(data) < 3:
            self.publish_status(
                "ERROR: format /arm/target_xyz = [x, y, z, duration_optional]"
            )
            return

        x = float(data[0])
        y = float(data[1])
        z = float(data[2])
        duration = float(data[3]) if len(data) >= 4 else 2.0

        def worker():
            with self.lock:
                try:
                    self.publish_status(
                        f"MOVE XYZ | X={x:.2f}, Y={y:.2f}, "
                        f"Z={z:.2f}, Duration={duration:.2f}"
                    )

                    q, pitch_auto, branch = self.arm.move_xyz(x, y, z, duration)

                    # ====================================================
                    # AUTO GRIPPER CLOSE
                    # Setelah robot sampai di koordinat tujuan,
                    # gripper langsung menutup otomatis.
                    # Setelah barang terjepit, wrist kembali ke 0 deg
                    # agar gripper lurus sejajar link wrist.
                    # ====================================================
                    if not self.arm.stop_request:
                        time.sleep(0.15)  # jeda kecil agar servo arm stabil dulu
                        self.arm.gripper_close()

                    if not self.arm.stop_request:
                        time.sleep(0.15)
                        q = self.arm.straighten_wrist(duration=0.7)

                    pose = self.arm.current_pose()

                    self.publish_status(
                        f"MOVE OK + GRIPPER CLOSE + WRIST STRAIGHT | "
                        f"branch={branch}, "
                        f"pitch_grip={pitch_auto:.2f}, "
                        f"base={q['base']:.2f}, "
                        f"shoulder={q['shoulder']:.2f}, "
                        f"elbow={q['elbow']:.2f}, "
                        f"wrist={q['wrist']:.2f} | "
                        f"X={pose['x']:.2f}, Y={pose['y']:.2f}, "
                        f"Z={pose['z']:.2f}, Pitch={pose['pitch']:.2f}, "
                        f"Gripper={self.arm.gripper_current:.2f}"
                    )

                except Exception as e:
                    self.publish_status(f"ERROR MOVE: {e}")

        threading.Thread(target=worker, daemon=True).start()

    def pick_drop_xyz_callback(self, msg):
        data = list(msg.data)

        if len(data) < 8:
            self.publish_status(
                "ERROR: format /arm/pick_drop_xyz = "
                "[pick_x, pick_y, pick_z, pick_duration, "
                "drop_x, drop_y, drop_z, drop_duration]"
            )
            return

        pick_x = float(data[0])
        pick_y = float(data[1])
        pick_z = float(data[2])
        pick_duration = float(data[3])
        drop_x = float(data[4])
        drop_y = float(data[5])
        drop_z = float(data[6])
        drop_duration = float(data[7])

        def worker():
            with self.lock:
                try:
                    self.publish_status(
                        f"PICK DROP START | "
                        f"Pick X={pick_x:.2f}, Y={pick_y:.2f}, Z={pick_z:.2f}, "
                        f"Drop X={drop_x:.2f}, Y={drop_y:.2f}, Z={drop_z:.2f}"
                    )

                    q, pitch_pick, pick_branch = self.arm.move_xyz(
                        pick_x,
                        pick_y,
                        pick_z,
                        pick_duration,
                    )

                    if self.arm.stop_request:
                        self.publish_status("PICK DROP STOPPED AFTER PICK MOVE")
                        return

                    self.publish_status(
                        f"PICK POINT OK | branch={pick_branch}, "
                        f"pitch_grip={pitch_pick:.2f}, "
                        f"base={q['base']:.2f}, shoulder={q['shoulder']:.2f}, "
                        f"elbow={q['elbow']:.2f}, wrist={q['wrist']:.2f}"
                    )

                    time.sleep(0.15)
                    if self.arm.stop_request:
                        self.publish_status("PICK DROP STOPPED BEFORE GRIPPER CLOSE")
                        return

                    self.arm.gripper_close()
                    self.publish_status(
                        f"GRIPPER CLOSE OK | Gripper={self.arm.gripper_current:.2f}"
                    )

                    if self.arm.stop_request:
                        self.publish_status("PICK DROP STOPPED AFTER GRIPPER CLOSE")
                        return

                    time.sleep(0.15)
                    q = self.arm.straighten_wrist(duration=0.7)

                    if self.arm.stop_request:
                        self.publish_status("PICK DROP STOPPED AFTER WRIST STRAIGHT")
                        return

                    self.publish_status(
                        f"WRIST STRAIGHT OK | "
                        f"base={q['base']:.2f}, shoulder={q['shoulder']:.2f}, "
                        f"elbow={q['elbow']:.2f}, wrist={q['wrist']:.2f}"
                    )

                    q, pitch_drop, drop_branch = self.arm.move_xyz_wrist_straight(
                        drop_x,
                        drop_y,
                        drop_z,
                        drop_duration,
                    )

                    if self.arm.stop_request:
                        self.publish_status("PICK DROP STOPPED DURING DROP MOVE")
                        return

                    self.publish_status(
                        f"DROP POINT OK | branch={drop_branch}, "
                        f"pitch_drop={pitch_drop:.2f}, "
                        f"base={q['base']:.2f}, shoulder={q['shoulder']:.2f}, "
                        f"elbow={q['elbow']:.2f}, wrist={q['wrist']:.2f}"
                    )

                    time.sleep(0.15)
                    if self.arm.stop_request:
                        self.publish_status("PICK DROP STOPPED BEFORE GRIPPER OPEN")
                        return

                    self.arm.gripper_open()

                    if self.arm.stop_request:
                        self.publish_status("PICK DROP STOPPED AFTER GRIPPER OPEN")
                        return

                    self.publish_status("DROP DONE, GO HOME")
                    self.arm.home()

                    if self.arm.stop_request:
                        self.publish_status("PICK DROP STOPPED DURING HOME")
                        return

                    pose = self.arm.current_pose()

                    self.publish_status(
                        f"PICK DROP OK + GRIPPER OPEN + HOME | "
                        f"X={pose['x']:.2f}, Y={pose['y']:.2f}, "
                        f"Z={pose['z']:.2f}, Pitch={pose['pitch']:.2f}, "
                        f"Gripper={self.arm.gripper_current:.2f}"
                    )

                except Exception as e:
                    self.publish_status(f"ERROR PICK DROP: {e}")

        threading.Thread(target=worker, daemon=True).start()

    def gripper_callback(self, msg):
        with self.lock:
            try:
                angle = float(msg.data)
                self.arm.stop_request = False
                self.arm.move_gripper(angle, self.arm.gripper_close_duration)
                self.publish_status(
                    f"GRIPPER = {self.arm.gripper_current:.2f} | "
                    f"Duration={self.arm.gripper_close_duration:.2f}s"
                )
            except Exception as e:
                self.publish_status(f"ERROR GRIPPER: {e}")

    def home_callback(self, msg):
        def worker():
            with self.lock:
                try:
                    self.publish_status("GO HOME")
                    self.arm.home()

                    pose = self.arm.current_pose()

                    self.publish_status(
                        f"HOME OK | "
                        f"X={pose['x']:.2f}, Y={pose['y']:.2f}, "
                        f"Z={pose['z']:.2f}, Pitch={pose['pitch']:.2f}"
                    )

                except Exception as e:
                    self.publish_status(f"ERROR HOME: {e}")

        threading.Thread(target=worker, daemon=True).start()

    def stop_callback(self, msg):
        self.arm.stop()
        self.publish_status("STOP REQUESTED")


def main(args=None):
    rclpy.init(args=args)

    node = ArmDriverNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        try:
            node.arm.release_all()
        except Exception:
            pass

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
