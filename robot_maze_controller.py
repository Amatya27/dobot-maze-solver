#!/usr/bin/env python3
"""
Dobot Maze Solver - DUAL CALIBRATION PATH TRACING + GEMINI AI
✅ Uses calibration_final.json (homography from colored blocks)
✅ Uses robot_calibration.json (scale from ruler)
✅ Gemini AI for intelligent path verification
✅ Perfect pixel-to-robot conversion for lab verification

QUICK START:
  python robot_maze_controller.py --start green --camera 1

SETUP:
  pip install google-generativeai
  export GEMINI_API_KEY=<your key>
"""

import os
import sys
import json
import cv2
import numpy as np
import subprocess
import argparse
import csv
import time

# Dobot API
try:
    from pydobot import Dobot
    import serial.tools.list_ports
except ImportError:
    print("⚠️ Install: pip install pydobot")
    sys.exit(1)

# Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    print("⚠️ Gemini not installed: pip install google-generativeai")
    GEMINI_AVAILABLE = False

# ==================== CONFIGURATION ====================
CALIBRATION_HOMOGRAPHY = "calibration_final.json"
CALIBRATION_SCALE = "robot_calibration.json"
CAMERA_HOME_POSITION = (228.6, -9.4, 90.1, -2.4)
HOVER_HEIGHT = -43
CAMERA_INDEX = 1
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ==================== GEMINI VERIFICATION ====================
def verify_path_with_gemini(image_path, start_color):
    """Use Gemini to verify the maze solution is correct"""
    if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        print("   ⚠️ Gemini API not available (set GEMINI_API_KEY)")
        return None
    
    try:
        print("\n   🤖 [Gemini] Analyzing maze solution...")
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        
        # Read the image with the traced path
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        prompt = f"""You are a maze verification expert. Analyze this maze image:

1. Is there a green START marker? Where?
2. Is there a red END marker? Where?
3. Is there a GREEN PATH traced through the maze?
4. Does the path connect START to END without crossing walls?
5. Rate the solution quality: Excellent/Good/Fair/Poor
6. Any issues or improvements?

Be concise and professional."""
        
        response = model.generate_content([
            {"mime_type": "image/jpeg", "data": image_data},
            prompt
        ])
        
        print("\n   [Gemini Analysis]")
        print("   " + "-" * 60)
        for line in response.text.split("\n"):
            print(f"   {line}")
        print("   " + "-" * 60)
        
        return response.text
        
    except Exception as e:
        print(f"   ⚠️ Gemini analysis failed: {e}")
        return None


def get_maze_insights_from_gemini(image_path):
    """Use Gemini to provide insights about the maze structure"""
    if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        return None
    
    try:
        print("\n   🤖 [Gemini] Analyzing maze structure...")
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        prompt = """Analyze this maze:
1. Maze complexity (Simple/Medium/Hard)
2. Number of major corridors: ~?
3. Estimated solution path length: short/medium/long
4. Key decision points: ?
Summary in 1-2 sentences."""
        
        response = model.generate_content([
            {"mime_type": "image/jpeg", "data": image_data},
            prompt
        ])
        
        print(f"   {response.text}")
        return response.text
        
    except Exception as e:
        print(f"   ⚠️ Could not get insights: {e}")
        return None

# ==================== DOBOT CONNECTION ====================
def connect_dobot():
    """Connect to Dobot"""
    print("🔌 Connecting to Dobot...")
    available_ports = list(serial.tools.list_ports.comports())
    
    if not available_ports:
        raise RuntimeError("❌ No serial ports found")
    
    for port in available_ports:
        try:
            device = Dobot(port=port.device, verbose=False)
            print(f"✓ Connected on {port.device}")
            return device
        except:
            continue
    
    raise RuntimeError("❌ Failed to connect to Dobot")


def move_to_camera_position(device):
    """Move to camera position"""
    x, y, z, r = CAMERA_HOME_POSITION
    print(f"📷 Moving to camera position: ({x}, {y}, {z}, {r})...")
    device.move_to(x, y, z, r, wait=True)
    print("✓ At camera position")


# ==================== CALIBRATION OFFSETS ====================
# Measured from get_pose.py - difference between expected and actual position
X_OFFSET_MM = 15.19   # Robot is 15.19 mm to the LEFT, move RIGHT
Y_OFFSET_MM = -8.92   # Robot is 8.92 mm BELOW, move UP

# ==================== CALIBRATION ====================
def load_calibrations():
    """Load BOTH calibration files"""
    print("✅ Loading calibrations...")
    
    # Load homography calibration
    if not os.path.exists(CALIBRATION_HOMOGRAPHY):
        print(f"   ❌ Missing: {CALIBRATION_HOMOGRAPHY}")
        return None
    
    with open(CALIBRATION_HOMOGRAPHY, "r") as f:
        homo_calib = json.load(f)
    
    if 'homography_calibration' not in homo_calib:
        print(f"   ❌ Missing homography_calibration")
        return None
    
    # Load scale calibration
    if not os.path.exists(CALIBRATION_SCALE):
        print(f"   ❌ Missing: {CALIBRATION_SCALE}")
        return None
    
    with open(CALIBRATION_SCALE, "r") as f:
        scale_calib = json.load(f)
    
    if 'pixels_per_mm' not in scale_calib:
        print(f"   ❌ Missing pixels_per_mm")
        return None
    
    print(f"   ✓ Homography: ✓ (colored blocks)")
    print(f"   ✓ Scale: {scale_calib['pixels_per_mm']:.6f} px/mm (from ruler)")
    print(f"   ✓ Robot Offset: X={X_OFFSET_MM} mm, Y={Y_OFFSET_MM} mm")
    
    if GEMINI_AVAILABLE and GEMINI_API_KEY:
        print(f"   ✓ Gemini AI: ✓ (enabled for verification)")
    
    return {
        "homography_matrix": np.array(homo_calib['homography_calibration']['homography_matrix']),
        "pixels_per_mm": scale_calib['pixels_per_mm'],
        "homography_data": homo_calib,
        "scale_data": scale_calib
    }



# ==================== CAMERA ====================
def capture_image(save_path="maze_capture.jpg", camera_index=1):
    """Capture image from camera"""
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        raise RuntimeError(f"❌ Cannot open camera {camera_index}")
    
    print(f"\n📸 Camera {camera_index} - ")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.putText(frame, " ", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Camera Preview", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE
            cv2.imwrite(save_path, frame)
            print(f"✓ Saved: {save_path}")
            break
        elif key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            return None
    
    cap.release()
    cv2.destroyAllWindows()
    return save_path


# ==================== COLOR DETECTION ====================
def detect_color_center(image, color_name):
    """Detect colored marker"""
    COLOR_RANGES = {
        'red': [
            (np.array([0, 100, 100]), np.array([10, 255, 255])),
            (np.array([170, 100, 100]), np.array([180, 255, 255]))
        ],
        'green': [(np.array([40, 40, 40]), np.array([80, 255, 255]))],
    }
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    
    for lower, upper in COLOR_RANGES[color_name]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    
    if M['m00'] == 0:
        return None
    
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    
    return (cx, cy)


# ==================== COORDINATE TRANSFORMATION ====================
def pixel_to_robot(px, py, homography_matrix):
    """Convert pixel to robot coordinates using homography WITH CALIBRATION OFFSET"""
    pixel_point = np.array([[[px, py]]], dtype=np.float32)
    robot_point = cv2.perspectiveTransform(pixel_point, homography_matrix)[0][0]
    
    # Apply calibration offsets
    x_mm = float(robot_point[0]) + X_OFFSET_MM
    y_mm = float(robot_point[1]) + Y_OFFSET_MM
    
    return x_mm, y_mm



# ==================== MAZE SOLVING ====================
def solve_maze(maze_image_path, start_color):
    """Run maze solver"""
    print("\n🧩 Solving maze with hybrid solver...")
    
    result = subprocess.run(
        [sys.executable, "hybrid_maze_solver.py", maze_image_path, start_color],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    if result.returncode != 0:
        print(f"❌ Solver failed:\n{result.stderr}")
        return None
    
    print(result.stdout)
    
    # Load path from CSV
    if not os.path.exists("path.csv"):
        print("❌ path.csv not found!")
        return None
    
    path_data = []
    with open("path.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path_data.append({
                "step": int(row["step"]),
                "x_px": int(row["x_px"]),
                "y_px": int(row["y_px"]),
                "direction": row["direction"]
            })
    
    print(f"✓ Loaded {len(path_data)} waypoints from path.csv")
    return path_data


# ==================== MAIN WORKFLOW ====================
def navigate_maze(device, start_color, camera_index):
    """Main navigation workflow"""
    
    print("\n" + "="*70)
    print(f"🚀 MAZE PATH TRACING VERIFICATION")
    print(f"   Starting color: {start_color.upper()}")
    print("="*70)
    
    # ========== STEP 1: Load BOTH Calibrations ==========
    print("\n[1/9] 📋 Loading BOTH calibrations...")
    calibrations = load_calibrations()
    if not calibrations:
        return
    
    homography_matrix = calibrations['homography_matrix']
    
    # ========== STEP 2: Move to Camera ==========
    print("\n[2/9] 📷 Moving to camera position...")
    move_to_camera_position(device)
    
    # ========== STEP 3: Capture Maze ==========
    print("\n[3/9] 📸 Capturing maze image...")
    input("   Press ENTER when maze is ready...")
    
    maze_img_path = capture_image("maze_capture.jpg", camera_index)
    if not maze_img_path:
        print("❌ Capture cancelled")
        return
    
    maze_image = cv2.imread(maze_img_path)
    print(f"   Image: {maze_image.shape[1]}×{maze_image.shape[0]} pixels")
    
    # ========== STEP 4: Get Gemini Maze Insights ==========
    print("\n[4/9] 🤖 Analyzing maze with Gemini AI...")
    get_maze_insights_from_gemini(maze_img_path)
    
    # ========== STEP 5: Detect Start Marker ==========
    print("\n[5/9] 🎯 Detecting start marker...")
    start_pixel = detect_color_center(maze_image, start_color)
    
    if not start_pixel:
        print(f"   ❌ Could not detect {start_color} marker!")
        print(f"   Try: Better lighting, ensure {start_color} circle is visible")
        return
    
    print(f"   ✓ Found {start_color.upper()} at pixel: {start_pixel}")
    
    # ========== STEP 6: Transform to Robot Coords ==========
    print("\n[6/9] 🔄 Converting to robot coordinates...")
    start_robot_x, start_robot_y = pixel_to_robot(
        start_pixel[0], start_pixel[1], homography_matrix
    )
    print(f"   ✓ Robot coordinates: ({start_robot_x:.2f}, {start_robot_y:.2f}) mm")
    
    # ========== STEP 7: Move to Start ==========
    print("\n[7/9] 🚀 Moving to start marker...")
    try:
        device.move_to(start_robot_x, start_robot_y, HOVER_HEIGHT, 
                      CAMERA_HOME_POSITION[3], wait=True)
        print(f"   ✓ Reached start position!")
    except Exception as e:
        print(f"   ⚠️ Error: {e}")
        return
    
    # ========== STEP 8: Solve Maze ==========
    print("\n[8/9] 🧭 Solving maze...")
    path_data = solve_maze(maze_img_path, start_color)
    
    if not path_data:
        print("   ❌ Failed to solve maze")
        return
    
    # ========== Convert Waypoints ==========
    print("\n   Converting waypoints to robot coordinates...")
    robot_waypoints = []
    
    for i, wp in enumerate(path_data):
        try:
            robot_x, robot_y = pixel_to_robot(
                wp["x_px"], wp["y_px"], homography_matrix
            )
            
            robot_waypoints.append({
                "step": wp["step"],
                "pixel_x": wp["x_px"],
                "pixel_y": wp["y_px"],
                "robot_x": robot_x,
                "robot_y": robot_y,
                "z": HOVER_HEIGHT,
                "direction": wp["direction"]
            })
            
            # Show first, middle, and last
            if i == 0 or i == len(path_data) // 2 or i == len(path_data) - 1:
                print(f"     Step {wp['step']:3d}: px({wp['x_px']:4d},{wp['y_px']:4d}) → robot({robot_x:7.2f},{robot_y:7.2f})")
        
        except Exception as e:
            print(f"   ⚠️ Step {wp['step']}: {e}")
    
    print(f"   ✓ Ready: {len(robot_waypoints)} waypoints")
    
    # ========== STEP 9: Execute Path ==========
    print("\n[9/9] 🤖 EXECUTING MAZE PATH")
    print("="*70)
    
    input("\n⏸  Press ENTER to START path tracing...")
    print("\n🚀 Robot is tracing path...\n")
    
    success_count = 0
    failed_count = 0
    
    for i, wp in enumerate(robot_waypoints):
        try:
            device.move_to(
                wp['robot_x'], 
                wp['robot_y'], 
                wp['z'], 
                CAMERA_HOME_POSITION[3], 
                wait=False
            )
            
            success_count += 1
            
            # Print progress
            if (i + 1) % max(1, len(robot_waypoints) // 10) == 0:
                progress = int((i + 1) / len(robot_waypoints) * 100)
                print(f"   [{progress:3d}%] Step {wp['step']:3d} ✓")
            
            time.sleep(0.2)
            
        except Exception as e:
            failed_count += 1
            print(f"   ⚠️ Step {wp['step']}: {e}")
    
    print("\n" + "="*70)
    print("✅ PATH TRACING COMPLETE!")
    print(f"   Success: {success_count}/{len(robot_waypoints)}")
    print(f"   Failed: {failed_count}/{len(robot_waypoints)}")
    print("="*70)
    
    # ========== VERIFY WITH GEMINI ==========
    print("\n🤖 [Gemini] Verifying maze solution...")
    verify_path_with_gemini("solved_path.png", start_color)
    
    # ========== Return Home ==========
    print("\n🏠 Returning to camera position...")
    try:
        move_to_camera_position(device)
        print("✓ Home")
    except:
        print("⚠️ Could not return home")


# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser(
        description="Dobot Maze Path Tracing (Dual Calibration + Gemini AI)"
    )
    parser.add_argument("--start", choices=["green", "red"], required=True,
                       help="Starting marker color")
    parser.add_argument("--camera", type=int, default=1,
                       help="Camera index (default: 1)")
    
    args = parser.parse_args()
    
    # Connect to robot
    device = connect_dobot()
    
    try:
        navigate_maze(device, args.start, args.camera)
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        device.close()
        print("\n👋 Disconnected")


if __name__ == "__main__":
    main()
