#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray
import math
import time

class BrainNode(Node):
    def __init__(self):
        super().__init__('brain_node')
        self.sub = self.create_subscription(Point, '/target_bbox', self.bbox_callback, 10)
        self.pub = self.create_publisher(Float64MultiArray, '/servo_angles', 10)
        self.timer = self.create_timer(0.1, self.state_machine)
        
        # --- KALIBRASI EYE-TO-HAND (METER) ---
        # Posisi titik pusat kamera (u=0.5, v=0.5) relatif terhadap titik (0,0) robot
        self.cam_center_x = 0.25 # Kamera ada sejauh 25 cm di depan robot
        self.cam_center_y = 0.00 # Kamera sejajar sumbu Y robot
        
        # Lebar area yang bisa dilihat kamera di atas meja (Meter)
        self.cam_fov_width_x = 0.40  # Kamera bisa melihat bidang selebar 40cm (X)
        self.cam_fov_height_y = 0.30 # Kamera bisa melihat bidang sepanjang 30cm (Y)
        
        # --- PARAMETER ROBOT (METER) ---
        self.L1 = 0.10 # Base ke Bahu
        self.L2 = 0.12 # Bahu ke Siku
        self.L3 = 0.12 # Siku ke Pergelangan
        self.L4 = 0.10 # Pergelangan ke Ujung Capit
        
        # State Machine
        self.state = "IDLE"
        self.target_x = 0.0
        self.target_y = 0.0
        self.drop_x = 0.15
        self.drop_y = 0.20
        self.last_seen = time.time()
        
        self.gripper_open = 45.0
        self.gripper_close = 120.0
        
        self.get_logger().info("Brain Node Siap. Menunggu deteksi objek...")

    def bbox_callback(self, msg):
        if msg.x >= 0: # Jika ada deteksi
            # Konversi Pinhole Sederhana (Normalisasi -> Meter Dunia Nyata)
            # Asumsi Z meja konstan (0.0)
            self.target_x = self.cam_center_x + (msg.x - 0.5) * self.cam_fov_width_x
            self.target_y = self.cam_center_y + (msg.y - 0.5) * self.cam_fov_height_y
            self.last_seen = time.time()
            
            if self.state == "IDLE":
                self.state = "APPROACH"

    def calculate_ik(self, x, y, z):
        # IK 4DOF Planar (Wrist selalu menghadap lurus ke bawah meja / Pitch -90 deg)
        pitch = math.radians(-90)
        
        theta1 = math.atan2(y, x) # Sudut Base
        
        # Hitung posisi pergelangan
        r = math.sqrt(x**2 + y**2)
        rw = r - self.L4 * math.cos(pitch)
        zw = z - self.L4 * math.sin(pitch) - self.L1
        
        # Aturan Cosinus untuk Bahu dan Siku
        D = (rw**2 + zw**2 - self.L2**2 - self.L3**2) / (2 * self.L2 * self.L3)
        D = max(-1.0, min(1.0, D))
        
        theta3 = math.atan2(-math.sqrt(1 - D**2), D) # Elbow down
        theta2 = math.atan2(zw, rw) - math.atan2(self.L3 * math.sin(theta3), self.L2 + self.L3 * math.cos(theta3))
        theta4 = pitch - theta2 - theta3
        
        # Convert ke derajat (dengan offset agar 90 derajat = lurus ke atas)
        return [
            math.degrees(theta1) + 90, 
            math.degrees(theta2) + 90, 
            math.degrees(theta3) + 90, 
            math.degrees(theta4) + 90
        ]

    def move_robot(self, x, y, z, gripper):
        angles = self.calculate_ik(x, y, z)
        msg = Float64MultiArray()
        msg.data = angles + [float(gripper)]
        self.pub.publish(msg)

    def state_machine(self):
        # Keamanan: Jika 2 detik objek hilang, kembali standby
        if time.time() - self.last_seen > 2.0 and self.state in ["IDLE", "APPROACH"]:
            self.state = "IDLE"
            self.move_robot(0.15, 0.0, 0.15, self.gripper_open) # Posisi Standby
            return

        if self.state == "IDLE":
            pass
            
        elif self.state == "APPROACH":
            self.get_logger().info("Target Terkunci! Menuju ke atas objek...")
            self.move_robot(self.target_x, self.target_y, 0.15, self.gripper_open) # Z = 15cm (Hover)
            time.sleep(1.5)
            self.state = "PICK"
            
        elif self.state == "PICK":
            self.get_logger().info("Mengambil objek...")
            self.move_robot(self.target_x, self.target_y, 0.02, self.gripper_open) # Z = 2cm (Turun)
            time.sleep(1.0)
            self.move_robot(self.target_x, self.target_y, 0.02, self.gripper_close) # Jepit
            time.sleep(0.5)
            self.move_robot(self.target_x, self.target_y, 0.15, self.gripper_close) # Angkat
            time.sleep(1.0)
            self.state = "PLACE"
            
        elif self.state == "PLACE":
            self.get_logger().info("Memindahkan ke kotak...")
            self.move_robot(self.drop_x, self.drop_y, 0.15, self.gripper_close) # Pindah
            time.sleep(2.0)
            self.move_robot(self.drop_x, self.drop_y, 0.05, self.gripper_close) # Turun dikit
            time.sleep(1.0)
            self.move_robot(self.drop_x, self.drop_y, 0.05, self.gripper_open) # Lepas
            time.sleep(0.5)
            self.state = "IDLE"

def main(args=None):
    rclpy.init(args=args)
    node = BrainNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
