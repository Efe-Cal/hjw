import subprocess
import cv2
import numpy as np
import os
import json


def closeness_to_center(img,detection):
    _,_,cx,cy = detection
    img_cx, img_cy = img.shape[1]//2, img.shape[0]//2
    return np.sqrt((cx - img_cx)**2 + (cy - img_cy)**2)

# ----------------- version 3 -----------------

MORPH_KERNEL = (5, 5)  # Kernel size for morphological operations

def isolate_and_subtract_channel(img, channel='r'):
    
    if img is None:
        raise ValueError("Image not found or path is incorrect.")

    channels = {'b': 0, 'g': 1, 'r': 2}
    if channel not in channels:
        raise ValueError("Channel must be 'r', 'g', or 'b'.")

    idx = channels[channel]
    other_idxs = [i for i in range(3) if i != idx]

    # Subtract other channels from the selected channel
    result_channel = img[:, :, idx]
    for oi in other_idxs:
        result_channel = cv2.subtract(result_channel, img[:, :, oi])

    # Create output image with only the result in the selected channel
    out_img = np.zeros_like(img)
    out_img[:, :, idx] = result_channel
    return out_img

# Example usage:
channels = ['r', 'g', 'b']
channel_names = {'r': 'Red', 'g': 'Green', 'b': 'Blue'}
img_path = './image.png'

def crop_image(img, x, y, w, h):
    cropped_img = img[y:y+h, x:x+w]
    return cropped_img

def merge_close_contours(contours, d_thresh=20):
    """
    Merge contours whose minimum point-to-point distance <= d_thresh (pixels).
    Returns a list of merged contours (convex hulls).
    """
    if not contours:
        return []

    n = len(contours)
    rects = [cv2.boundingRect(c) for c in contours]  # (x,y,w,h)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pb] = pa

    def bbox_dist(r1, r2):
        x1,y1,w1,h1 = r1
        x2,y2,w2,h2 = r2
        x_gap = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
        y_gap = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
        return (x_gap**2 + y_gap**2)**0.5

    def contour_min_dist(c1, c2):
        p1 = c1.reshape(-1, 2).astype(np.float32)
        p2 = c2.reshape(-1, 2).astype(np.float32)
        # vectorized pairwise distances
        d = np.sqrt(((p1[:, None, :] - p2[None, :, :]) ** 2).sum(axis=2))
        return float(d.min())

    # build connectivity (fast bbox filter, then exact distance)
    for i in range(n):
        for j in range(i + 1, n):
            if bbox_dist(rects[i], rects[j]) > d_thresh:
                continue
            if contour_min_dist(contours[i], contours[j]) <= d_thresh:
                union(i, j)

    # group and merge
    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    merged = []
    for idxs in groups.values():
        pts = np.vstack([contours[k].reshape(-1, 2) for k in idxs])
        hull = cv2.convexHull(pts.astype(np.int32))
        merged.append(hull)

    return merged

def is_contour_closer_to_red_or_yellow(img, contour):

    # Create a mask for the contour
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, [contour], 255)
    
    # Convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Calculate the mean HSV values within the contour
    mean_hsv = cv2.mean(hsv, mask=mask)
    mean_h, mean_s, mean_v = mean_hsv[:3]

    red_ranges = [
        (0, 15),    # Lower red range
        (160, 180)  # Upper red range
    ]
    yellow_range = (20, 30)
    
    # Calculate distance to red (considering wraparound)
    red_distance = float('inf')
    for red_min, red_max in red_ranges:
        if red_min <= mean_h <= red_max:
            red_distance = 0  # Direct hit
            break
        else:
            # Calculate minimum distance to this range
            dist_to_range = min(abs(mean_h - red_min), abs(mean_h - red_max))
            red_distance = min(red_distance, dist_to_range)
    
    # Handle hue wraparound for red (0-180 scale)
    if mean_h > 90:  # If hue is in upper half, also check distance to 0
        wraparound_distance = min(abs(mean_h - 180), abs(mean_h - 0))
        if wraparound_distance < 10:  # Within red range considering wraparound
            red_distance = min(red_distance, wraparound_distance)
    
    # Calculate distance to yellow
    yellow_min, yellow_max = yellow_range
    if yellow_min <= mean_h <= yellow_max:
        yellow_distance = 0
    else:
        yellow_distance = min(abs(mean_h - yellow_min), abs(mean_h - yellow_max))
    
    # Return the closer color
    if red_distance <= yellow_distance:
        return 'r'
    else:
        return 'y'

def detect_and_extract_contours(img_path):
    img = cv2.imread(img_path)
    img = crop_image(img, 1735, 657, 172, 122)
    # cv2.imshow('Cropped Image', img)
    detections = []
    for ch in channels:
        result = isolate_and_subtract_channel(img, ch)

        # Extract the relevant channel as a single-channel image for contour detection
        channel_idx = {"r": 2, "g": 1, "b": 0}[ch]
        single_channel = result[:, :, channel_idx]
        # cv2.imshow(f'{channel_names[ch]} Channel', single_channel)

        _, single_channel_thresh = cv2.threshold(single_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological operations
        morph_open = cv2.morphologyEx(single_channel_thresh, cv2.MORPH_OPEN, np.ones(MORPH_KERNEL, np.uint8))
        closed = cv2.morphologyEx(morph_open, cv2.MORPH_CLOSE, np.ones(MORPH_KERNEL, np.uint8))

        # Convert closed image to BGR for colored drawing
        result_bgr = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)

        contours, _ = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        contours = merge_close_contours(contours, d_thresh=50)
        # print(f"{channel_names[ch]} channel - Contours after merging: {len(contours)}")
        
        cv2.drawContours(result_bgr, contours, -1, (255, 0, 0), 1)  # Draw contours in blue
        
        # Debug: Print contour areas
        # for cnt in contours:
        #     area = cv2.contourArea(cnt)
        #     print(f"Contour area in {channel_names[ch]} channel: {area}")

        # Filter contours based on area
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 4000]
        # print(f"{channel_names[ch]} channel - Contours after area filtering: {len(contours)}")
        # Filter contours based on aspect ratio
        ratio = 2
        contours = [cnt for cnt in contours if cv2.boundingRect(cnt)[2] / cv2.boundingRect(cnt)[3] < ratio]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # cv2.drawContours(result_bgr, contours, -1, (0, 255, 255), 2)  # Highlight brightest contour in yellow
        # cv2.imshow(f'Isolated {channel_names[ch]} Channel', result_bgr)
        if len(contours) > 0:
            M = cv2.moments(contours[0])
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        
            if ch=="r":
                color = is_contour_closer_to_red_or_yellow(img, contours[0])
                detections.append((color,cx,cy))
            else:
                detections.append((ch,cx,cy))
    
    if len(detections) == 1:
        return detections[0][0][0]
    elif len(detections) > 1:
        # Choose the detection closest to the image center
        distances = [closeness_to_center(img, d) for d in detections]
        best_idx = np.argmin(distances)
        return detections[best_idx][0][0]
        
    print("Nothing detected, defaulting to blue")
    return None

# ----------------- version 1.2 -----------------

MORPHOLOGY_KERNEL_SIZE = (7, 7)  # Kernel size for morphological operations
DIST_TRESH = 0.4  # Distance threshold for distance transform
EXTENSION_OFFSET = (10, 30, 30)  # Offset for extending color ranges

def load_config():
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, 'config.json')
    if not os.path.exists(config_path):
        print(f"Configuration file not found at {config_path}. Using the hardcoded default configuration.")
        return {"big_box_crop": [1735, 657, 172, 122], "color_ranges": {"red": [[[0, 143, 54], [12, 253, 164]], [[162, 143, 54], [179, 253, 164]]], "green": [[[60, 137, 13], [90, 247, 123]]], "blue": [[[94, 173, 45], [124, 255, 155]]], "yellow": [[[7, 170, 99], [37, 255, 209]]]}}
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def build_clean_mask(hsv: np.ndarray,
                     ranges: list[tuple[list[int],tuple[int]]],
                     kernel_size: tuple[int,int]=MORPHOLOGY_KERNEL_SIZE) -> np.ndarray:
    """Build and clean mask for a list of HSV ranges."""
    mask = None
    for lo, hi in ranges:
        part = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = part if mask is None else cv2.bitwise_or(mask, part)
    kernel = np.ones(kernel_size, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def extend_color_range(color_range: list, offset:tuple=(5,10,10)) -> list:
    """Extends a color range by an offset."""
    if isinstance(color_range[0][0],list):
        lo = extend_color_range(color_range[0], offset)[0]
        hi = extend_color_range(color_range[1], offset)[0]
    else:
        lo, hi = color_range
        lo = list(max(0, c - o) for c, o in zip(lo, offset))
        hi = list(min(255, c + o) for c, o in zip(hi, offset))
        
    return [[lo, hi]]

retry_with_extended = False
def detect_boxes(img,color_ranges):
    """
    Detects red, blue, yellow, and green boxes in the image and returns their order from left to right.
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    detections = []
    for color, ranges in color_ranges.items():
        # Create mask for the color
        mask = build_clean_mask(hsv, ranges, kernel_size=MORPHOLOGY_KERNEL_SIZE)

        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform,DIST_TRESH*dist_transform.max(),255,0)

        # Find contours
        sure_fg = sure_fg.astype(np.uint8) 
        contours, _ = cv2.findContours(dist_transform.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # for cnt in contours:
        contour_areas = [cv2.contourArea(cnt) for cnt in contours if cv2.contourArea(cnt) > 4000]

        area = sum(contour_areas)
        if area < 500:
            if not area == 0:
                print(f"Area too small: {area}   {color}")
            continue
        
        cnt = contours[np.argmax(contour_areas)]  # Get the largest contour
        
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    

        detections.append((color,area,cx,cy))

    
    if len(detections) == 0 and retry_with_extended==False:
        print("No boxes detected")
        return detect_boxes(img, extend_color_range(color_ranges))
    if len(detections) == 0 and retry_with_extended==True:
        print("No boxes detected even after extending ranges")
        return None 
    elif len(detections) == 1:
        return detections[0][0][0]    
    elif len(detections) > 1:
        # Normalize area and closeness to center, then combine equally
        areas = np.array([d[1] for d in detections])
        centers = np.array([closeness_to_center(img,d) for d in detections])
        norm_areas = (areas - areas.min()) / (areas.ptp() if areas.ptp() > 0 else 1)
        norm_centers = (centers - centers.min()) / (centers.ptp() if centers.ptp() > 0 else 1)
        scores = norm_areas + (1 - norm_centers)  # larger area and closer to center preferred
        best_idx = np.argmax(scores)
        detection = detections[best_idx][0][0]
        return detection

def crop_image(img, x:int, y:int, w:int, h:int) -> np.ndarray:
    return img[y:y+h, x:x+w]
    

def get_box_color(img_path:str,display:bool=False) -> str:
    config = load_config()
    color_ranges = config["color_ranges"]
    
    image = cv2.imread(img_path)
    
    box_image = crop_image(image, *config["big_box_crop"])

    d = detect_boxes(box_image, color_ranges, display)
    return d


def decider(v1_2, v3):
    if v1_2 == v3:
        return v1_2
    if v1_2 in ["r","y"] and v3 in ["r","y"]:
        return v1_2
    if v3 == "g":
        return 'g'
    if v1_2 == None:
        return v3
    if v3 == None:
        return v1_2
    if v1_2 == None and v3 == None:
        return "b"

def image_job():
    # rpicam-still --output ./image.png --timeout 200 --width 1920 --height 1080 --rotation 180
    subprocess.run(["rpicam-still", "--output", img_path, "--timeout", "200", "--width", "1920", "--height", "1080", "--rotation", "180"])
    
    v1_2 = get_box_color(img_path)
    v3 = detect_and_extract_contours(img_path)
    
    print(f"v1.2 detected: {v1_2}, v3 detected: {v3}")

    return decider(v1_2, v3)


