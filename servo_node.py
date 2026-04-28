import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray
import math
import time

class OtakNode(Node):
    def __init__(self):
        super().__init__('otak_node')
        self.sub = self.create_subscription(Point, '/target_bbox', self.deteksi_masuk, 10)
        self.pub = self.create_publisher(Float64MultiArray, '/sudut_servo', 10)
        self.timer = self.create_timer(0.1, self.logika_robot)
        
        # --- KALIBRASI KAMERA EYE-TO-HAND (METER) ---
        self.cam_offset_x = 0.20 # Jarak X dari tiang base robot ke tengah pandangan kamera
        self.cam_offset_y = 0.00 
        self.fov_x = 0.30        # Lebar pandangan kamera di atas meja
        self.fov_y = 0.20
        
        # --- PANJANG LENGAN (METER) ---
        # WAJIB: Ukur ulang pakai penggaris dari titik pusat engsel ke engsel!
        self.L1 = 0.10 # Tiang base ke engsel bahu bawah
        self.L2 = 0.12 # Engsel bahu ke engsel siku (Bagian orange vertikal)
        self.L3 = 0.15 # Engsel siku ke engsel pergelangan (Bagian putih horizontal)
        self.L4 = 0.12 # Engsel pergelangan ke ujung tempat capit menjepit barang
        
        self.state = "IDLE"
        self.tx, self.ty = 0.0, 0.0
        self.last_seen = time.time()
        print("OTAK READY: Menunggu deteksi objek...")

    def deteksi_masuk(self, msg):
        if msg.x >= 0:
            self.tx = self.cam_offset_x + (msg.x - 0.5) * self.fov_x
            self.ty = self.cam_offset_y + (msg.y - 0.5) * self.fov_y
            self.last_seen = time.time()
            if self.state == "IDLE":
                self.state = "APPROACH"

    def hitung_ik(self, x, y, z):
        # 1. Base Angle
        theta1 = math.atan2(y, x)
        
        # Kita ingin capit selalu menghadap lurus ke BAWAH saat ngambil barang
        pitch = math.radians(-90)
        
        # 2. Radius Jarak
        r = math.sqrt(x**2 + y**2)
        rw = r - self.L4 * math.cos(pitch)
        zw = z - self.L4 * math.sin(pitch) - self.L1
        
        # 3. IK untuk 2-Link (Bahu & Siku)
        c3 = (rw**2 + zw**2 - self.L2**2 - self.L3**2) / (2 * self.L2 * self.L3)
        c3 = max(-1.0, min(1.0, c3)) # Limit biar ga error out-of-reach
        
        theta3 = math.atan2(-math.sqrt(1 - c3**2), c3) # Set siku menekuk ke bawah
        theta2 = math.atan2(zw, rw) - math.atan2(self.L3 * math.sin(theta3), self.L2 + self.L3 * math.cos(theta3))
        
        # 4. Sudut pergelangan relatif thd Siku
        theta4 = pitch - theta2 - theta3
        
        # --- MAPPING KE FISIK BERDASARKAN FOTO 90 DERAJAT LUU ---
        t1_deg = math.degrees(theta1)
        t2_deg = math.degrees(theta2)
        t3_deg = math.degrees(theta3)
        t4_deg = math.degrees(theta4)

        servo_base = t1_deg + 90.0
        servo_shoulder = t2_deg           # 90 geometri = 90 tegak di fisik
        servo_elbow = t3_deg + 180.0      # -90 geometri = 90 lurus datar di fisik
        servo_wrist = t4_deg + 90.0       # 0 geometri = 90 lurus sejajar di fisik
        
        # CATATAN: Kalau saat jalan ternyata lengan bahu (shoulder) malah muter 
        # ke belakang meja, hapus tanda pagar (#) di baris bawah ini:
        # servo_shoulder = 180.0 - servo_shoulder

        return [servo_base, servo_shoulder, servo_elbow, servo_wrist]

    def gerak(self, x, y, z, jepit):
        sudut = self.hitung_ik(x, y, z)
        msg = Float64MultiArray(data=sudut + [float(jepit)])
        self.pub.publish(msg)

    def logika_robot(self):
        if time.time() - self.last_seen > 2.0 and self.state in ["IDLE", "APPROACH"]:
            self.state = "IDLE"
            self.gerak(0.15, 0.0, 0.15, 45.0) # Posisi aman / jepit buka
            return

        if self.state == "APPROACH":
            print(f"Target di X:{self.tx:.2f}, Y:{self.ty:.2f}. Mendekat...")
            self.gerak(self.tx, self.ty, 0.15, 45.0) # Hover 15cm
            time.sleep(1.5)
            self.state = "PICK"
            
        elif self.state == "PICK":
            print("Ngambil...")
            self.gerak(self.tx, self.ty, 0.02, 45.0) # Turun ke 2cm
            time.sleep(1.0)
            self.gerak(self.tx, self.ty, 0.02, 120.0) # Jepit
            time.sleep(0.5)
            self.gerak(self.tx, self.ty, 0.15, 120.0) # Angkat kembali ke 15cm
            time.sleep(1.0)
            self.state = "PLACE"
            
        elif self.state == "PLACE":
            print("Pindah ke keranjang...")
            self.gerak(0.10, 0.20, 0.15, 120.0) # Koordinat lokasi keranjang
            time.sleep(2.0)
            self.gerak(0.10, 0.20, 0.05, 45.0) # Turun & Buka Capit
            time.sleep(1.0)
            self.gerak(0.10, 0.20, 0.15, 45.0) # Naik lagi
            time.sleep(1.0)
            self.state = "IDLE"

def main():
    rclpy.init()
    node = OtakNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
