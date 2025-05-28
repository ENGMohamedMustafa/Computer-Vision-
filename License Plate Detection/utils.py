import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import matplotlib.pyplot as plt
import re
model = YOLO("D:/Python Sofwares/env/yolov8x.pt")
reader = easyocr.Reader(['en'], gpu=True)

def detect_license_plates(image_path):
    """
    Complete license plate detection function
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read image")
        return [], None
    
    # Make a copy for annotation
    annotated_image = image.copy()
    
    # List to store detected license plate texts
    license_plate_texts = []
    
    # Run YOLO detection
    results = model.predict(image)
    
    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        names = results[0].names
        
        print(f"Found {len(boxes)} detections")
        
        for i, box in enumerate(boxes):
            try:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                label = names[class_id]
                
                print(f"Detection {i}: {label} ({confidence:.2f}) at [{x1},{y1},{x2},{y2}]")
                
                # Check if detected object is a vehicle
                vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
                if label in vehicle_classes and confidence > 0.4:
                    
                    # Extract vehicle region
                    vehicle_image = image[y1:y2, x1:x2]
                    
                    if vehicle_image.size > 0:
                        # Look for license plates in the vehicle region
                        plate_regions = find_license_plate_regions(vehicle_image)
                        
                        for px1, py1, px2, py2 in plate_regions:
                            # Convert to absolute coordinates
                            abs_x1 = x1 + px1
                            abs_y1 = y1 + py1
                            abs_x2 = x1 + px2
                            abs_y2 = y1 + py2
                            
                            # Extract license plate region
                            license_plate_image = image[abs_y1:abs_y2, abs_x1:abs_x2]
                            
                            if license_plate_image.size > 0:
                                # Process the license plate
                                plate_text = process_license_plate(license_plate_image)
                                
                                if plate_text:
                                    license_plate_texts.append(plate_text)
                                    print(f"Detected license plate text: {plate_text}")
                                    
                                    # Draw rectangle around the detected license plate
                                    cv2.rectangle(annotated_image, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 255, 0), 2)
                                    
                                    # Add text label with background
                                    text_size = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                                    cv2.rectangle(annotated_image, (abs_x1, abs_y1 - 25), 
                                                (abs_x1 + text_size[0], abs_y1), (0, 255, 0), -1)
                                    cv2.putText(annotated_image, plate_text, (abs_x1, abs_y1 - 5), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            except Exception as e:
                print(f"Error processing detection {i}: {e}")
                continue
    
    return license_plate_texts, annotated_image

def find_license_plate_regions(vehicle_image):
    """
    Find potential license plate regions within a vehicle image
    """
    if vehicle_image is None or vehicle_image.size == 0:
        return []
    
    height, width = vehicle_image.shape[:2]
    
    # Focus on bottom 60% of vehicle where license plates are typically located
    roi_start = int(height * 0.4)
    roi = vehicle_image[roi_start:height, 0:width]
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    plate_candidates = []
    
    for contour in contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate aspect ratio and area
        if h > 0:
            aspect_ratio = w / h
            area = w * h
            
            # Filter based on typical license plate characteristics
            # Aspect ratio: 2:1 to 5:1, minimum area, reasonable dimensions
            if (2.0 <= aspect_ratio <= 5.0 and 
                area >= 600 and 
                w >= 40 and h >= 10):
                
                # Adjust coordinates back to vehicle image
                adjusted_y = roi_start + y
                plate_candidates.append((x, adjusted_y, x + w, adjusted_y + h))
    
    return plate_candidates

def process_license_plate(license_plate_image):
    """
    Process license plate image and extract text
    """
    try:
        # Convert to grayscale
        license_plate_gray = cv2.cvtColor(license_plate_image, cv2.COLOR_BGR2GRAY)
        
        # Resize if too small for better OCR
        height, width = license_plate_gray.shape
        if height < 40 or width < 120:
            scale_factor = max(120/width, 40/height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            license_plate_gray = cv2.resize(license_plate_gray, (new_width, new_height), 
                                          interpolation=cv2.INTER_CUBIC)
        
        # Apply preprocessing
        # Method 1: Adaptive thresholding
        license_plate_thresh = cv2.adaptiveThreshold(
            license_plate_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Method 2: OTSU thresholding (alternative)
        _, license_plate_thresh = cv2.threshold(
            license_plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Perform OCR
        plate_text_results = reader.readtext(license_plate_thresh, detail=0, paragraph=False)
        
        if not plate_text_results:
            return None
        
        # Combine all detected text
        combined_text = ' '.join(plate_text_results)
        
        # Clean and validate the text
        plate_candidates = []
        
        # Split by common separators and process each part
        text_parts = re.split(r'[-\s]+', combined_text)
        
        for part in text_parts:
            cleaned = part.replace("-", "").replace(" ", "").upper()
            
            # Check if it matches license plate pattern
            # At least 3 characters, max 10, contains both letters and numbers
            if (re.fullmatch(r"[A-Z0-9]{3,10}", cleaned) and 
                re.search(r"[A-Z]", cleaned) and 
                re.search(r"\d", cleaned)):
                plate_candidates.append(cleaned)
        
        # Also check the combined text
        cleaned_combined = combined_text.replace("-", "").replace(" ", "").upper()
        if (re.fullmatch(r"[A-Z0-9]{3,10}", cleaned_combined) and 
            re.search(r"[A-Z]", cleaned_combined) and 
            re.search(r"\d", cleaned_combined)):
            plate_candidates.append(cleaned_combined)
        
        # Return the best candidate (longest valid text)
        if plate_candidates:
            # Remove duplicates and sort by length
            unique_candidates = list(set(plate_candidates))
            best_candidate = max(unique_candidates, key=len)
            return best_candidate
    
    except Exception as e:
        print(f"Error processing license plate: {e}")
    
    return None