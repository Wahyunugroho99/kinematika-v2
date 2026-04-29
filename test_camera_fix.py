#!/usr/bin/env python3
"""
Simple test to verify camera opening fix works on Linux
"""

import sys
import cv2

def build_camera_sources(camera_id: int):
    return [camera_id]


def build_capture_backends():
    if not sys.platform.startswith('linux'):
        return [cv2.CAP_ANY]

    backends = []
    for attr_name in ('CAP_ANY', 'CAP_V4L2'):
        backend = getattr(cv2, attr_name, None)
        if backend is not None and backend not in backends:
            backends.append(backend)
    return backends or [cv2.CAP_ANY]


def capture_is_usable(cap) -> bool:
    for _ in range(3):
        ok, frame = cap.read()
        if ok and frame is not None and getattr(frame, 'size', 0) > 0:
            return True
    return False


def test_camera_opening():
    """Test that camera can be opened with our fix logic"""
    print("Testing camera opening logic...")

    # Simulate the logic we added to real_camera_node.py
    camera_id = 0  # Default camera ID
    cap = None
    sources_to_try = build_camera_sources(camera_id)
    backends_to_try = build_capture_backends()

    last_error = None
    for source in sources_to_try:
        try:
            for backend in backends_to_try:
                test_cap = cv2.VideoCapture(source, backend)
                if test_cap.isOpened():
                    test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    if not capture_is_usable(test_cap):
                        test_cap.release()
                        continue
                    cap = test_cap
                    camera_id = source
                    break
                test_cap.release()

            if cap is not None:
                break

        except Exception as e:
            last_error = e
            continue

    if cap is None or not cap.isOpened():
        tried_sources = ', '.join(str(src) for src in sources_to_try)
        error_msg = f"Cannot open camera index {camera_id} (tried: {tried_sources})"
        if last_error:
            error_msg += f": {last_error}"
        if sys.platform.startswith('linux'):
            error_msg += "\nOn Linux, you may need to:\n" \
                       "  1. Check if your user is in the 'video' group: groups\n" \
                       "  2. If not, add yourself: sudo usermod -a -G video $USER\n" \
                       "  3. Log out and back in, or reboot\n" \
                       "  4. Ensure no other process is using the camera"
        print(f"FAILED: {error_msg}")
        return False
    else:
        print(f"SUCCESS: Camera {camera_id} opened successfully")
        # Test reading a frame
        ret, frame = cap.read()
        if ret:
            print(f"SUCCESS: Frame captured - shape: {frame.shape}")
        else:
            print("WARNING: Could not read frame (but camera opened)")
        cap.release()
        return True

if __name__ == "__main__":
    success = test_camera_opening()
    sys.exit(0 if success else 1)
