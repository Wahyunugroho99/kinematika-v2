# ⚡ CYBER ARM // Eye-to-Hand YOLO Robot Arm

<p align="center">
  <b>NEON VISION • ROS 2 CONTROL • AUTO GRIP • 4-DOF ROBOT ARM</b>
</p>

<p align="center">
  <img alt="ROS2" src="https://img.shields.io/badge/ROS%202-robotics-00f5ff?style=for-the-badge&logo=ros&logoColor=white">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.x-ff00c8?style=for-the-badge&logo=python&logoColor=white">
  <img alt="YOLO" src="https://img.shields.io/badge/YOLO-Eye--to--Hand-b6ff00?style=for-the-badge">
  <img alt="PCA9685" src="https://img.shields.io/badge/PCA9685-Servo%20Driver-7f00ff?style=for-the-badge">
</p>

```text
╔══════════════════════════════════════════════════════════════╗
║  C Y B E R   A R M   C O N T R O L   S Y S T E M           ║
║  Target detected → XYZ mapped → IK solved → gripper locked  ║
╚══════════════════════════════════════════════════════════════╝
```

## 🧬 Overview

**CYBER ARM** adalah sistem kontrol robot arm berbasis **ROS 2**, **Python**, **YOLO**, dan **eye-to-hand calibration**. Project ini menggabungkan kontrol servo 4-DOF + gripper, GUI input XYZ, visualisasi posisi, kamera YOLO, serta transformasi koordinat kamera ke koordinat robot memakai homography.

Sistem ini cocok untuk eksperimen **pick-and-place**, deteksi objek berbasis kamera, dan kontrol robot arm edukatif dengan servo PWM via **PCA9685**.

---

## ✨ Core Features

- 🤖 **Robot Arm Driver** untuk 4-DOF + gripper.
- 🎯 **Input XYZ** tanpa input pitch manual.
- 🧠 **Inverse Kinematics otomatis** dengan mode gripper menghadap bawah saat menuju target.
- 🦾 **Auto grip** setelah target pick tercapai.
- 📦 **Pick + Drop workflow** dalam satu command.
- 👁️ **YOLO eye-to-hand camera node** untuk deteksi objek real-time.
- 🧭 **Kalibrasi homography** dari pixel kamera ke koordinat robot.
- 🖥️ **GUI Tkinter** dengan camera view, detection table, top view, dan back view.
- 📡 Komunikasi penuh via topic ROS 2.

---

## 🗂️ Project Structure

```text
.
├── arm_driver_auto_grip.py          # ROS 2 node driver robot arm + IK + servo PCA9685
├── arm_xyz_input_auto_grip.py       # GUI kontrol XYZ, pick/drop, visualisasi, kalibrasi kamera
├── camera_yolo_eye_to_hand.py       # ROS 2 node kamera + YOLO detection publisher
├── eye_to_hand_calibration.json     # Data kalibrasi image_points, robot_points, homography
└── README.md                        # Dokumentasi project
```

---

## 🕹️ ROS 2 Nodes

### 1. `arm_driver_auto_grip.py`

Node utama untuk menggerakkan robot arm.

**Fungsi utama:**

- Inisialisasi PCA9685 pada 50 Hz.
- Mapping model angle ke servo angle.
- Forward kinematics dan inverse kinematics.
- Gerak halus memakai `smoothstep`.
- Auto close gripper setelah target XYZ tercapai.
- Mode pick-drop otomatis.

**Subscribe:**

| Topic | Type | Format |
|---|---|---|
| `/arm/target_xyz` | `Float64MultiArray` | `[x, y, z, duration]` |
| `/arm/pick_drop_xyz` | `Float64MultiArray` | `[pick_x, pick_y, pick_z, pick_duration, drop_x, drop_y, drop_z, drop_duration]` |
| `/arm/gripper` | `Float64` | `angle` |
| `/arm/home` | `Empty` | Home command |
| `/arm/stop` | `Empty` | Stop command |

**Publish:**

| Topic | Type | Format |
|---|---|---|
| `/arm/current_xyz` | `Float64MultiArray` | `[x, y, z, pitch]` |
| `/arm/servo_angle` | `Float64MultiArray` | `[base, shoulder, elbow, wrist, gripper]` |
| `/arm/status` | `String` | Status text |

---

### 2. `arm_xyz_input_auto_grip.py`

GUI cyber-control center untuk input target, drop point, visualisasi posisi arm, kamera, dan deteksi objek.

**Fungsi utama:**

- Input manual `X Y Z` untuk pick.
- Input `Drop X Y Z`.
- Tombol `Pick + Drop`, `Pick Saja`, `Home`, `Stop`, `Gripper Buka`, `Gripper Tutup`.
- Visualisasi top view `(X-Y)` dan back view `(X-Z)`.
- Menampilkan feed kamera dari `/vision/image_annotated`.
- Membaca detection dari `/vision/detections`.
- Konversi pixel kamera ke koordinat robot memakai homography.
- Kalibrasi 4 titik langsung dari GUI.

---

### 3. `camera_yolo_eye_to_hand.py`

Node kamera untuk inference YOLO dan publish hasil deteksi.

**Parameter:**

| Parameter | Default | Deskripsi |
|---|---:|---|
| `model_path` | `best.pt` | Path model YOLO |
| `camera_source` | `2` | Index kamera atau path stream/video |
| `confidence` | `0.8` | Confidence threshold YOLO |
| `publish_hz` | `15.0` | Frekuensi publish frame dan detection |

**Publish:**

| Topic | Type | Isi |
|---|---|---|
| `/vision/image_annotated` | `sensor_msgs/Image` | Frame kamera dengan bounding box YOLO |
| `/vision/detections` | `String` JSON | Data bbox, confidence, class, center |
| `/vision/status` | `String` | Status kamera/model |

---

## 🧠 Kinematics Configuration

Konfigurasi link arm:

| Link | Panjang |
|---|---:|
| `L1` | `8 cm` |
| `L2` | `12 cm` |
| `L3` | `8 cm` |
| `L4` | `15 cm` |

Sistem koordinat:

```text
X = kiri / kanan robot
Y = depan robot
Z = atas robot
```

Home model angle:

```text
base     = 0°
shoulder = 90°
elbow    = -90°
wrist    = -90° / 0° tergantung mode node
```

> Catatan: sesuaikan nilai kalibrasi servo, arah servo, trim, serta limit servo di `arm_driver_auto_grip.py` agar aman untuk mekanik robot kamu.

---

## 👁️ Eye-to-Hand Calibration

File `eye_to_hand_calibration.json` menyimpan:

```json
{
  "image_points": [[...], [...], [...], [...]],
  "robot_points": [[...], [...], [...], [...]],
  "homography": [[...], [...], [...]]
}
```

Alur kalibrasi:

1. Jalankan kamera YOLO dan GUI.
2. Aktifkan **Mode klik** pada panel kalibrasi.
3. Klik 4 titik pada gambar kamera.
4. Isi koordinat robot untuk 4 titik tersebut.
5. Klik **Simpan Kalibrasi**.
6. GUI akan memakai homography untuk mengubah pixel objek menjadi koordinat robot `X,Y`.

---

## ⚙️ Requirements

### Hardware

- Robot arm 4-DOF + gripper.
- Servo motor untuk base, shoulder, elbow, wrist, gripper.
- PCA9685 servo driver.
- Kamera USB / CSI / stream.
- Board yang mendukung I2C, contoh Raspberry Pi.

### Software

- ROS 2.
- Python 3.
- OpenCV.
- NumPy.
- Pillow.
- Tkinter.
- Ultralytics YOLO.
- `cv_bridge`.
- Adafruit PCA9685 dan servo library.

Contoh instalasi Python dependency:

```bash
pip install opencv-python numpy pillow ultralytics adafruit-circuitpython-pca9685 adafruit-circuitpython-motor
```

> Untuk ROS 2 dependency seperti `rclpy`, `sensor_msgs`, `std_msgs`, dan `cv_bridge`, gunakan instalasi sesuai distro ROS 2 yang kamu pakai.

---

## 🚀 Run the System

### 1. Jalankan driver robot arm

```bash
python3 arm_driver_auto_grip.py
```

### 2. Jalankan kamera YOLO

```bash
python3 camera_yolo_eye_to_hand.py
```

Dengan parameter custom:

```bash
python3 camera_yolo_eye_to_hand.py --ros-args \
  -p model_path:=best.pt \
  -p camera_source:=2 \
  -p confidence:=0.8 \
  -p publish_hz:=15.0
```

### 3. Jalankan GUI kontrol

```bash
python3 arm_xyz_input_auto_grip.py
```

---

## 📡 Command Examples

### Move ke target XYZ

```bash
ros2 topic pub /arm/target_xyz std_msgs/msg/Float64MultiArray \
"{data: [0.0, 24.0, 22.0, 2.0]}"
```

### Pick + Drop

```bash
ros2 topic pub /arm/pick_drop_xyz std_msgs/msg/Float64MultiArray \
"{data: [0.0, 24.0, 12.0, 2.0, 10.0, 24.0, 18.0, 2.0]}"
```

### Buka gripper

```bash
ros2 topic pub /arm/gripper std_msgs/msg/Float64 "{data: 30.0}"
```

### Tutup gripper

```bash
ros2 topic pub /arm/gripper std_msgs/msg/Float64 "{data: 120.0}"
```

### Home

```bash
ros2 topic pub /arm/home std_msgs/msg/Empty "{}"
```

### Stop

```bash
ros2 topic pub /arm/stop std_msgs/msg/Empty "{}"
```

---

## 🧪 Suggested Workflow

```text
[BOOT]
  ↓
Run arm driver
  ↓
Run camera YOLO node
  ↓
Run GUI
  ↓
Check camera feed + detection table
  ↓
Calibrate 4 eye-to-hand points
  ↓
Select detected object
  ↓
Click Pick Objek Terdeteksi
  ↓
Robot moves → grips → drops
```

---

## 🛡️ Safety Notes

- Pastikan servo tidak mentok secara mekanik sebelum menjalankan command otomatis.
- Uji gerakan dengan durasi lambat terlebih dahulu.
- Selalu siapkan akses ke `/arm/stop`.
- Jangan menaruh tangan di area kerja robot saat auto grip aktif.
- Kalibrasi ulang jika posisi kamera berubah.
- Turunkan confidence YOLO hanya jika false detection masih aman untuk mekanik.

---

## 🧯 Troubleshooting

| Problem | Kemungkinan Penyebab | Solusi |
|---|---|---|
| Kamera tidak terbuka | `camera_source` salah | Coba `0`, `1`, `2`, atau path stream |
| YOLO error | `best.pt` tidak ditemukan / model invalid | Pastikan path model benar |
| Koordinat objek meleset | Kalibrasi kamera berubah | Ulangi kalibrasi 4 titik |
| Servo bergerak terbalik | `direction` salah | Ubah `direction` pada joint calibration |
| Target tidak tercapai | Di luar workspace IK | Pilih titik XYZ yang lebih dekat/aman |
| Gripper terlalu kuat/lemah | Angle close kurang sesuai | Sesuaikan nilai `gripper_close()` |

---

## 🧩 Customization

### Ubah model YOLO

```bash
python3 camera_yolo_eye_to_hand.py --ros-args -p model_path:=runs/detect/train/weights/best.pt
```

### Ubah confidence

```bash
python3 camera_yolo_eye_to_hand.py --ros-args -p confidence:=0.6
```

### Ubah source kamera

```bash
python3 camera_yolo_eye_to_hand.py --ros-args -p camera_source:=0
```

---

## 🛰️ Topic Map

```text
┌──────────────────────────────┐
│ camera_yolo_eye_to_hand.py   │
│  /vision/image_annotated     │────┐
│  /vision/detections          │────┤
│  /vision/status              │────┤
└──────────────────────────────┘    │
                                    ▼
┌──────────────────────────────┐    │
│ arm_xyz_input_auto_grip.py   │◄───┘
│  GUI + calibration + command │
│  /arm/target_xyz             │────┐
│  /arm/pick_drop_xyz          │────┤
│  /arm/gripper                │────┤
│  /arm/home                   │────┤
│  /arm/stop                   │────┤
└──────────────────────────────┘    │
                                    ▼
┌──────────────────────────────┐
│ arm_driver_auto_grip.py      │
│  IK + Servo + Auto Grip      │
│  /arm/current_xyz            │────┐
│  /arm/servo_angle            │────┤
│  /arm/status                 │────┘
└──────────────────────────────┘
```

---

## 🌌 Cyberpunk Mission Log

```text
STATUS      : ONLINE
VISION      : SCANNING
KINEMATICS  : SOLVING
GRIPPER     : ARMED
NEON CORE   : READY
```

> Build your arm. Calibrate the eye. Let the machine reach into neon.

---

## 📜 License

Tambahkan license sesuai kebutuhan project kamu, misalnya MIT, Apache-2.0, atau private internal use.
