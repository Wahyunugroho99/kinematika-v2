#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

class ServoNode(Node):
    def __init__(self):
        super().__init__('servo_node')
        self.sub = self.create_subscription(Float64MultiArray, '/servo_angles', self.servo_cb, 10)
        
        # Setup I2C untuk Raspberry Pi 5
        i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(i2c)
        self.pca.frequency = 50
        
        # Setup 5 Servo (Ch 0,1,2,3,4) dengan standar pulse MG996R
        self.servos = [servo.Servo(self.pca.channels[i], min_pulse=600, max_pulse=2400) for i in range(5)]
        
        self.get_logger().info("Hardware PCA9685 Aktif. Siap menerima perintah...")

    def servo_cb(self, msg):
        angles = msg.data # Urutan: [Base, Shoulder, Elbow, Wrist, Gripper]
        for i, angle in enumerate(angles):
            if i < len(self.servos):
                # Proteksi batas aman mekanik servo (0 - 180 derajat)
                safe_angle = max(0.0, min(180.0, angle))
                
                # ---------------------------------------------------------
                # BLOK KOREKSI ARAH SERVO (REVERSE DIRECTION)
                # ---------------------------------------------------------
                # Index 0 = Base
                # Index 1 = Shoulder (Bahu) <-- Ini yang kita balik
                # Index 2 = Elbow (Siku)
                # Index 3 = Wrist (Pergelangan)
                # Index 4 = Gripper (Capit)
                
                if i == 1:
                    # Balik arah putaran servo bahu
                    safe_angle = 180.0 - safe_angle
                
                # Kalau misalnya siku (Elbow) lu ikutan kebalik juga, 
                # hapus tanda pagar (#) di bawah ini:
                # if i == 2:
                #     safe_angle = 180.0 - safe_angle
                # ---------------------------------------------------------

                try:
                    self.servos[i].angle = safe_angle
                except Exception as e:
                    self.get_logger().error(f"Error putar servo {i}: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ServoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Sangat Penting: Matikan tegangan servo saat program di-stop (Ctrl+C)
        node.pca.deinit() 
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
