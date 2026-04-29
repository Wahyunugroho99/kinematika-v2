#!/usr/bin/env python3
"""
Simple test to verify camera opening fix works on Linux
"""

import sys
import cv2

def test_camera_opening():
    """Test that camera can be opened with our fix logic"""
    print("Testing camera opening logic...")

    # Simulate the logic we added to real_camera_node.py
    camera_id = 0  # Default camera ID
    cap = None
    camera_indices_to_try = [camera_id]

    # On Linux, try additional common camera indices
    if sys.platform.startswith('linux'):
        camera_indices_to_try.extend([0, 1, 2, 3])

    # Remove duplicates while preserving order
    seen = set()
    unique_indices = []
    for idx in camera_indices_to_try:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)

    last_error = None
    for idx in unique_indices:
        try:
            # Try different backends for better Linux compatibility
            backends = [cv2.CAP_ANY]
            if sys.platform.startswith('linux'):
                backends.extend([cv2.CAP_V4L2, cv2.CAP_GSTREAMER])

            for backend in backends:
                test_cap = cv2.VideoCapture(idx, backend)
                if test_cap.isOpened():
                    cap = test_cap
                    camera_id = idx  # Update to the working index
                    break

            if cap is not None:
                break

        except Exception as e:
            last_error = e
            continue

    if cap is None or not cap.isOpened():
        error_msg = f"Cannot open camera index {camera_id}"
        if last_error:
            error_msg += f": {last_error}"
        if sys.platform.startswith('linux'):
            error_msg += "\nOn Linux, you may need to:\n" \
                       "  1. Check if your user is in the 'video' group: groups\n" \
                       "  2. If not, add yourself: sudo usermod -a -G video $USER\n" \
                       "  3. Log out and back in, or reboot\n" \
                       "  4. Try different camera indices (0,1,2,3)\n" \
                       "  5. Ensure no other process is using the camera"
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