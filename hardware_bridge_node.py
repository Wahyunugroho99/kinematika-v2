#!/usr/bin/env python3
"""
hardware_bridge_node.py
Node ini menjembatani ROS2 dengan perangkat keras nyata (PCA9685).
Menerima sudut (radian) dari /joint_states dan memutar servo fisik.
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

# Import library hardware Adafruit
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

class HardwareBridgeNode(Node):
    def __init__(self):
        super().__init__('hardware_bridge_node')
        
        # Inisialisasi I2C dan PCA9685
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(self.i2c)
        self.pca.frequency = 50 # Standar frekuensi servo RC adalah 50Hz
        
        # Mapping Channel PCA9685 ke Joint Robot
        # Sesuaikan dengan colokan di perangkat keras Anda
        # Format: { 'joint_index': (pca_channel, min_pulse, max_pulse, offset_derajat) }
        # min/max_pulse standar servo adalah 500-2500, tapi MG996R kadang butuh 600-2400.
        self.servos = {
            0: {"servo": servo.Servo(self.pca.channels[0], min_pulse=600, max_pulse=2400), "dir": 1,  "offset": 90}, # Base
            1: {"servo": servo.Servo(self.pca.channels[1], min_pulse=600, max_pulse=2400), "dir": -1, "offset": 90}, # Shoulder (dibalik)
            2: {"servo": servo.Servo(self.pca.channels[2], min_pulse=600, max_pulse=2400), "dir": 1,  "offset": 90}, # Elbow
            3: {"servo": servo.Servo(self.pca.channels[3], min_pulse=600, max_pulse=2400), "dir": 1,  "offset": 90}, # Wrist
}
        self.clamp_active = {joint_idx: False for joint_idx in self.servos}
        self.unsupported_joint_warned = False
        
        # Channel khusus Gripper (Capit)
        self.gripper_channel = 4
        self.gripper_servo = servo.Servo(self.pca.channels[self.gripper_channel], min_pulse=600, max_pulse=2400)
        
        # Subscribers
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.cmd_sub = self.create_subscription(String, '/robot_command', self.cmd_cb, 10)
        self.hw_joint_pub = self.create_publisher(JointState, '/joint_states_hw', 10)
        
        self.get_logger().info("Hardware Bridge PCA9685 Aktif. Siap memutar servo!")

    def joint_cb(self, msg):
        """Menerima array Radian dari IBVS Controller dan mengubahnya ke Derajat."""
        positions_rad = msg.position
        applied_positions = [0.0] * len(positions_rad)

        unsupported_indices = [i for i in range(len(positions_rad)) if i not in self.servos]
        if unsupported_indices and not self.unsupported_joint_warned:
            self.get_logger().warning(
                f"Joint {unsupported_indices} tidak punya mapping servo. Viz hardware akan menahan joint tersebut di 0 rad."
            )
            self.unsupported_joint_warned = True
        
        for i, rad in enumerate(positions_rad):
            if i in self.servos:
                # Konversi: 0 radian = 90 derajat (Tengah)
                # Rumus: (rad * 180 / pi) + 90
                cfg = self.servos[i]
                raw_deg = cfg["dir"] * math.degrees(rad) + cfg["offset"]
                
                # Batasi (Clamp) agar tidak merusak servo mekanis
                deg = max(0.0, min(180.0, raw_deg))
                is_clamped = abs(deg - raw_deg) > 1e-6
                if is_clamped and not self.clamp_active[i]:
                    self.get_logger().warning(
                        f"Servo joint {i} kena clamp: request {raw_deg:.1f}°, applied {deg:.1f}°."
                    )
                    self.clamp_active[i] = True
                elif not is_clamped and self.clamp_active[i]:
                    self.get_logger().info(f"Servo joint {i} kembali dalam range aman.")
                    self.clamp_active[i] = False

                applied_positions[i] = math.radians((deg - cfg["offset"]) / cfg["dir"])
                try:
                    cfg["servo"].angle = deg
                except Exception as e:
                    self.get_logger().error(f"Error memutar servo {i}: {e}")

        hw_msg = JointState()
        hw_msg.position = applied_positions
        self.hw_joint_pub.publish(hw_msg)

    def cmd_cb(self, msg):
        """Mendengarkan perintah capit."""
        cmd = msg.data
        if cmd == "ATTACH_OBJECT":
            # Tutup capit (Sesuaikan sudutnya dengan mekanik capit Anda)
            self.gripper_servo.angle = 120.0 
            self.get_logger().info("Gripper: DITUTUP")
        elif cmd == "DETACH_OBJECT":
            # Buka capit
            self.gripper_servo.angle = 45.0
            self.get_logger().info("Gripper: DIBUKA")

    def destroy_node(self):
        """Matikan semua servo (Release torque) saat node dimatikan."""
        for i in self.servos:
            self.servos[i]["servo"].angle = None
        self.gripper_servo.angle = None
        self.pca.deinit()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = HardwareBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
