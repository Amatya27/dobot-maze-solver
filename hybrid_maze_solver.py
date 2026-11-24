#!/usr/bin/env python3
"""
Maze solver: finds the start/end markers, builds a safe corridor away from
the walls, plans a path with A* (falls back to skeleton BFS / plain BFS) and
cuts it down to a few waypoints with Ramer-Douglas-Peucker.

Usage:
  python hybrid_maze_solver.py maze.jpg green
"""

import sys
import cv2
import numpy as np
from collections import deque
import heapq
import csv
import os

# -*- coding: utf-8 -*-
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ==================== GEMINI AI SETUP ====================
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ Gemini not installed: pip install google-generativeai")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# ========================================================

# ==================== CONFIGURATION ====================
MIN_CLEARANCE_FROM_WALL_PX = 15
CENTER_BIAS = 25.0
PATH_THICKNESS = 2
WAYPOINT_EPSILON = 15

COLOR_RANGES = {
    'red': [
        (np.array([0, 80, 80]), np.array([10, 255, 255])),
        (np.array([170, 80, 80]), np.array([180, 255, 255]))
    ],
    'green': [(np.array([40, 40, 40]), np.array([80, 255, 255]))],
    'yellow': [(np.array([20, 100, 100]), np.array([30, 255, 255]))],
    'blue': [(np.array([100, 100, 100]), np.array([130, 255, 255]))]
}

# ==================== COLOR DETECTION ====================
def detect_color_center(image, color_name):
    """Centroid of the largest blob of the given color"""
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

def find_markers_robust(bgr, start_color):
    """Find markers - works BOTH directions"""
    print(f"\n🎯 Detecting markers (START: {start_color.upper()})...")
    
    start_pt = detect_color_center(bgr, start_color)
    if start_pt is None:
        raise RuntimeError(f"Could not detect START {start_color} marker")
    
    end_color = 'red' if start_color == 'green' else 'green'
    
    end_pt = detect_color_center(bgr, end_color)
    if end_pt is None:
        raise RuntimeError(f"Could not detect END {end_color} marker")
    
    print(f" ✓ START ({start_color.upper()}): {start_pt}")
    print(f" ✓ END   ({end_color.upper()}): {end_pt}")
    
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    color_union = np.zeros(hsv.shape[:2], dtype=np.uint8)
    
    for color in [start_color, end_color]:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in COLOR_RANGES[color]:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        color_union = cv2.bitwise_or(color_union, mask)
    
    return start_pt, end_pt, color_union

# ==================== WALL DETECTION ====================
def threshold_walls(bgr, color_mask_to_clear):
    """Better wall detection"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    _, walls = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    if color_mask_to_clear is not None:
        walls[color_mask_to_clear > 0] = 0
    
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    walls = cv2.morphologyEx(walls, cv2.MORPH_OPEN, k, 1)
    walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, k, 2)
    
    return walls

def inside_open_mask(walls, frame=2):
    """Extract open space"""
    H, W = walls.shape
    closed = walls.copy()
    cv2.rectangle(closed, (0, 0), (W-1, H-1), 255, frame)
    
    open_tmp = cv2.bitwise_not(closed)
    outside = open_tmp.copy()
    mask = np.zeros((H+2, W+2), np.uint8)
    cv2.floodFill(outside, mask, (0, 0), 128)
    
    outside_mask = np.zeros_like(open_tmp)
    outside_mask[outside == 128] = 255
    
    inside = cv2.subtract(open_tmp, outside_mask)
    
    return inside

def erode_by_px(mask255, px):
    """Erode mask"""
    if px <= 0:
        return (mask255 > 0).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*px+1, 2*px+1))
    return cv2.erode((mask255 > 0).astype(np.uint8) * 255, k, iterations=1)

def skeletonize(bin255):
    """Skeletonization"""
    img = (bin255 > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    while True:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    
    return skel

# ==================== PATHFINDING ====================
def nearest_nonzero(img, pt):
    """Find nearest white pixel"""
    x, y = map(int, pt)
    H, W = img.shape
    
    if 0 <= x < W and 0 <= y < H and img[y, x] > 0:
        return (x, y)
    
    for r in range(1, max(H, W), 1):
        xmin, xmax = max(0, x-r), min(W-1, x+r)
        ymin, ymax = max(0, y-r), min(H-1, y+r)
        roi = img[ymin:ymax+1, xmin:xmax+1]
        
        nz = cv2.findNonZero(roi)
        if nz is not None:
            nz = nz.reshape(-1, 2)
            d2 = (nz[:, 0] + xmin - x)**2 + (nz[:, 1] + ymin - y)**2
            idx = int(np.argmin(d2))
            return (int(nz[idx, 0] + xmin), int(nz[idx, 1] + ymin))
    
    return (x, y)

def astar_centered(allow255, dist_norm, start, end, center_w=CENTER_BIAS):
    """A* with center bias"""
    h, w = allow255.shape
    sx, sy = start
    ex, ey = end
    
    if not (0 <= sx < w and 0 <= sy < h and 0 <= ex < w and 0 <= ey < h):
        return []
    if allow255[sy, sx] == 0 or allow255[ey, ex] == 0:
        return []
    
    INF = 1e12
    g = np.full((h, w), INF, dtype=np.float32)
    parent = np.full((h, w, 2), -1, dtype=np.int32)
    
    def heuristic(x, y):
        return float(abs(x - ex) + abs(y - ey))
    
    g[sy, sx] = 0.0
    open_heap = [(heuristic(sx, sy), 0.0, sx, sy)]
    closed = np.zeros((h, w), np.uint8)
    neighbors = [(0, -1), (0, 1), (1, 0), (-1, 0)]
    
    while open_heap:
        f_curr, g_curr, x, y = heapq.heappop(open_heap)
        
        if closed[y, x]:
            continue
        closed[y, x] = 1
        
        if (x, y) == (ex, ey):
            break
        
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < w and 0 <= ny < h and allow255[ny, nx] > 0 and not closed[ny, nx]:
                step_cost = 1.0 + center_w * (1.0 - float(dist_norm[ny, nx]))
                tentative = g_curr + step_cost
                
                if tentative < g[ny, nx]:
                    g[ny, nx] = tentative
                    parent[ny, nx] = [x, y]
                    f = tentative + heuristic(nx, ny)
                    heapq.heappush(open_heap, (f, tentative, nx, ny))
    
    if parent[ey, ex, 0] == -1 and (ex, ey) != (sx, sy):
        return []
    
    path = [(ex, ey)]
    x, y = ex, ey
    
    while (x, y) != (sx, sy):
        px, py = parent[y, x]
        if px == -1:
            break
        path.append((int(px), int(py)))
        x, y = int(px), int(py)
    
    path.append((sx, sy))
    path.reverse()
    
    return path

def bfs_path(binary255, start, end):
    """BFS pathfinding"""
    h, w = binary255.shape
    sx, sy = start
    ex, ey = end
    
    if not (0 <= sx < w and 0 <= sy < h and 0 <= ex < w and 0 <= ey < h):
        return []
    if binary255[sy, sx] == 0 or binary255[ey, ex] == 0:
        return []
    
    visited = np.zeros((h, w), np.uint8)
    parent = np.full((h, w, 2), -1, dtype=np.int32)
    dq = deque([(sx, sy)])
    visited[sy, sx] = 1
    neighbors = [(0, -1), (0, 1), (1, 0), (-1, 0)]
    
    while dq:
        x, y = dq.popleft()
        
        if (x, y) == (ex, ey):
            break
        
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < w and 0 <= ny < h and visited[ny, nx] == 0 and binary255[ny, nx] > 0:
                visited[ny, nx] = 1
                parent[ny, nx] = [x, y]
                dq.append((nx, ny))
    
    if parent[ey, ex, 0] == -1 and (ex, ey) != (sx, sy):
        return []
    
    path = [(ex, ey)]
    x, y = ex, ey
    
    while (x, y) != (sx, sy):
        px, py = parent[y, x]
        if px == -1:
            break
        path.append((int(px), int(py)))
        x, y = int(px), int(py)
    
    path.append((sx, sy))
    path.reverse()
    
    return path

# ==================== WAYPOINT REDUCTION ====================
def reduce_waypoints_simple(path, epsilon=WAYPOINT_EPSILON):
    """Simple but effective waypoint reduction"""
    if len(path) <= 2:
        return path
    
    def perpendicular_distance(point, line_start, line_end):
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        if x1 == x2 and y1 == y2:
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        
        numerator = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
        denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        
        return numerator / denominator
    
    def rdp(points, start, end, eps):
        if end - start <= 1:
            return [start, end]
        
        max_dist = 0
        max_idx = start
        
        for i in range(start + 1, end):
            d = perpendicular_distance(points[i], points[start], points[end])
            if d > max_dist:
                max_dist = d
                max_idx = i
        
        if max_dist > eps:
            left = rdp(points, start, max_idx, eps)
            right = rdp(points, max_idx, end, eps)
            return left[:-1] + right
        else:
            return [start, end]
    
    indices = rdp(path, 0, len(path) - 1, epsilon)
    return [path[i] for i in sorted(set(indices))]

# ==================== PATH PLANNING ====================
def plan_path(bgr, start_color):
    """Main pathfinding"""
    start_pt, end_pt, color_union = find_markers_robust(bgr, start_color)
    
    print(f"\n🔄 Building corridor...")
    walls = threshold_walls(bgr, color_union)
    inside = inside_open_mask(walls, frame=2)
    
    print(f"  Open space: {cv2.countNonZero(inside)} pixels")
    
    dist = cv2.distanceTransform(inside, cv2.DIST_L2, 3).astype(np.float32)
    dmax = float(dist.max()) if dist.max() > 0 else 1.0
    dist_norm = dist / (dmax + 1e-6)
    
    for margin_px in range(MIN_CLEARANCE_FROM_WALL_PX, -1, -1):
        print(f"\n🔍 Trying margin={margin_px}px...")
        safe = erode_by_px(inside, margin_px)
        
        safe_count = cv2.countNonZero(safe)
        if safe_count < 50:
            print(f"  ⚠️  Too small ({safe_count} pixels)")
            continue
        
        start_safe = nearest_nonzero(safe, start_pt)
        end_safe = nearest_nonzero(safe, end_pt)
        
        print(f"  Safe pixels: {safe_count}")
        
        path = astar_centered(safe, dist_norm, start_safe, end_safe, center_w=CENTER_BIAS)
        
        if path:
            print(f"  ✓ A* found: {len(path)} points")
            return path, start_pt, end_pt
        
        skel = skeletonize(safe)
        if cv2.countNonZero(skel) > 0:
            start_skel = nearest_nonzero(skel, start_safe)
            end_skel = nearest_nonzero(skel, end_safe)
            path = bfs_path(skel, start_skel, end_skel)
            
            if path:
                print(f"  ✓ Skeleton BFS found: {len(path)} points")
                return path, start_pt, end_pt
        
        path = bfs_path(safe, start_safe, end_safe)
        
        if path:
            print(f"  ✓ BFS found: {len(path)} points")
            return path, start_pt, end_pt
    
    raise RuntimeError("No path found!")

# ==================== OUTPUT ====================
def save_path_csv(path, output_csv="path.csv"):
    """Save CSV"""
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'x_px', 'y_px', 'direction'])
        
        for i, (x, y) in enumerate(path):
            x, y = int(x), int(y)
            if i == 0:
                direction = "START"
            elif i == len(path) - 1:
                direction = "END"
            else:
                dx = path[i+1][0] - x
                dy = path[i+1][1] - y
                
                if abs(dx) > abs(dy):
                    direction = "RIGHT" if dx > 0 else "LEFT"
                else:
                    direction = "DOWN" if dy > 0 else "UP"
            
            writer.writerow([i, x, y, direction])
    
    print(f"✓ Saved: {output_csv}")

def visualize_path(bgr, path, start_pt, end_pt, output_img="solved_path.png"):
    """Visualize path"""
    vis = bgr.copy()
    
    if len(path) > 1:
        pts = np.array([(int(p[0]), int(p[1])) for p in path], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], isClosed=False, color=(0, 255, 0),
                     thickness=PATH_THICKNESS, lineType=cv2.LINE_AA)
    
    cv2.circle(vis, start_pt, 12, (0, 255, 0), -1)
    cv2.circle(vis, end_pt, 12, (0, 0, 255), -1)
    
    for i, (x, y) in enumerate(path):
        if i > 0 and i < len(path) - 1:
            cv2.circle(vis, (int(x), int(y)), 5, (255, 0, 255), -1)
    
    cv2.imwrite(output_img, vis)
    print(f"✓ Saved: {output_img}")

# ==================== MAIN ====================
def main():
    if len(sys.argv) < 3:
        print("Usage: python hybrid_maze_solver.py <image> <color>")
        print("Example: python hybrid_maze_solver.py maze.jpg green")
        sys.exit(1)
    
    image_path = sys.argv[1]
    start_color = sys.argv[2].lower()
    
    if start_color not in ('green', 'red'):
        print("Error: color must be 'green' or 'red'")
        sys.exit(1)
    
    print(f"\n📸 Loading: {image_path}")
    bgr = cv2.imread(image_path)
    
    if bgr is None:
        print(f"Error: Cannot load {image_path}")
        sys.exit(1)
    
    print(f"✓ Size: {bgr.shape[1]}×{bgr.shape[0]}")
    
    print(f"\n🧭 Planning path...")
    try:
        path, start_pt, end_pt = plan_path(bgr, start_color)
        
        reduced_path = reduce_waypoints_simple(path, epsilon=WAYPOINT_EPSILON)
        print(f"✓ Reduced: {len(path)} → {len(reduced_path)} waypoints")
        
        save_path_csv(reduced_path, "path.csv")
        visualize_path(bgr, reduced_path, start_pt, end_pt, "solved_path.png")
        
        print(f"\n✅ SUCCESS! {len(reduced_path)} waypoints")
        
    except RuntimeError as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
