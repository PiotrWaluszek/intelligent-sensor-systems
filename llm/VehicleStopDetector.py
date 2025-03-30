import cv2
import numpy as np
import time
import os
from datetime import datetime

def detect_stopped_vehicles(video_path, output_folder='detected_vehicles'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Błąd: Nie można otworzyć pliku wideo.")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"FPS wideo: {fps}")
    print(f"Wymiary wideo: {frame_width}x{frame_height}")
    
    two_seconds_frames = int(fps * 2)
    print(f"Liczba klatek na 2 sekundy: {two_seconds_frames}")
    
    history = 300
    varThreshold = 25
    detectShadows = False
    
    backSub = cv2.createBackgroundSubtractorMOG2(history, varThreshold, detectShadows)
    
    min_contour_area = 1000
    min_stop_frames = two_seconds_frames
    vehicle_tracking = {}
    next_id = 0
    
    max_captures_per_vehicle = 1
    
    movement_threshold = 5
    
    stabilization_frames = min(5, int(fps) // 2)
    
    proximity_threshold = 0.6
    bottom_region_threshold = 0.5
    
    frame_count = 0
    
    last_save_time = time.time() - 10
    min_save_interval = 3
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Koniec pliku wideo lub błąd odczytu.")
            break
        
        frame_count += 1
        current_time = time.time()
        
        display_frame = frame.copy()
        
        fgMask = backSub.apply(frame)
        
        kernel = np.ones((5, 5), np.uint8)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for vehicle_id in vehicle_tracking:
            vehicle_tracking[vehicle_id]['visible'] = False
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_contour_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            center_x = x + w // 2
            center_y = y + h // 2
            
            matched = False
            for vehicle_id, vehicle_data in vehicle_tracking.items():
                prev_x, prev_y = vehicle_data['center']
                
                match_threshold = max(50, min(w, h) // 3)
                
                if abs(center_x - prev_x) < match_threshold and abs(center_y - prev_y) < match_threshold:
                    dx = abs(center_x - prev_x)
                    dy = abs(center_y - prev_y)
                    movement = np.sqrt(dx*dx + dy*dy)
                    
                    vehicle_tracking[vehicle_id]['center'] = (center_x, center_y)
                    vehicle_tracking[vehicle_id]['visible'] = True
                    vehicle_tracking[vehicle_id]['rect'] = (x, y, w, h)
                    vehicle_tracking[vehicle_id]['area'] = area
                    
                    is_wide = w / frame_width > proximity_threshold
                    is_tall = h / frame_height > proximity_threshold
                    is_bottom = (y + h) > (frame_height * bottom_region_threshold)
                    is_close = is_wide or is_tall or is_bottom
                    
                    vehicle_tracking[vehicle_id]['is_close'] = is_close
                    
                    vehicle_tracking[vehicle_id]['movement_history'].append(movement)
                    if len(vehicle_tracking[vehicle_id]['movement_history']) > stabilization_frames:
                        vehicle_tracking[vehicle_id]['movement_history'].pop(0)
                    
                    avg_movement = sum(vehicle_tracking[vehicle_id]['movement_history']) / len(vehicle_tracking[vehicle_id]['movement_history'])
                    
                    if avg_movement < movement_threshold:
                        vehicle_tracking[vehicle_id]['stop_frames'] += 1
                        
                        if (vehicle_tracking[vehicle_id]['stop_frames'] >= min_stop_frames and 
                            vehicle_tracking[vehicle_id]['is_close'] and
                            vehicle_tracking[vehicle_id]['captures'] < max_captures_per_vehicle and
                            current_time - last_save_time >= min_save_interval):
                            
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"{output_folder}/stopped_vehicle_{vehicle_id}_{timestamp}.jpg"
                            cv2.imwrite(filename, frame)
                            print(f"Zapisano klatkę z zatrzymanym pojazdem do: {filename}")
                            
                            vehicle_tracking[vehicle_id]['captures'] += 1
                            last_save_time = current_time
                    else:
                        if avg_movement > movement_threshold * 1.5:
                            vehicle_tracking[vehicle_id]['stop_frames'] = 0
                    
                    matched = True
                    break
            
            if not matched:
                is_wide = w / frame_width > proximity_threshold
                is_tall = h / frame_height > proximity_threshold
                is_bottom = (y + h) > (frame_height * bottom_region_threshold)
                is_close = is_wide or is_tall or is_bottom
                
                vehicle_tracking[next_id] = {
                    'center': (center_x, center_y),
                    'rect': (x, y, w, h),
                    'area': area,
                    'stop_frames': 0,
                    'visible': True,
                    'captures': 0,
                    'movement_history': [0] * stabilization_frames,
                    'is_close': is_close
                }
                next_id += 1
        
        vehicle_ids = list(vehicle_tracking.keys())
        for vehicle_id in vehicle_ids:
            if not vehicle_tracking[vehicle_id]['visible']:
                if 'invisible_count' not in vehicle_tracking[vehicle_id]:
                    vehicle_tracking[vehicle_id]['invisible_count'] = 1
                else:
                    vehicle_tracking[vehicle_id]['invisible_count'] += 1
                
                if vehicle_tracking[vehicle_id]['invisible_count'] > two_seconds_frames:
                    del vehicle_tracking[vehicle_id]
            else:
                vehicle_tracking[vehicle_id]['invisible_count'] = 0
        
        for vehicle_id, vehicle_data in vehicle_tracking.items():
            if not vehicle_data['visible']:
                continue
                
            x, y, w, h = vehicle_data['rect']
            
            if vehicle_data['is_close']:
                if vehicle_data['stop_frames'] > 0:
                    ratio = min(1.0, vehicle_data['stop_frames'] / min_stop_frames)
                    color = (0, int(255 * (1 - ratio)), 255)
                else:
                    color = (0, 255, 255)
            else:
                if vehicle_data['stop_frames'] > 0:
                    ratio = min(1.0, vehicle_data['stop_frames'] / min_stop_frames)
                    color = (int(255 * ratio), 255, 0)
                else:
                    color = (0, 255, 0)
            
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            
            info_text = f"ID: {vehicle_id}"
            cv2.putText(display_frame, info_text, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if vehicle_data['stop_frames'] > 0:
                percent = min(100, int(vehicle_data['stop_frames'] * 100 / min_stop_frames))
                stop_text = f"Stop: {percent}%"
                cv2.putText(display_frame, stop_text, (x, y + h + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                proximity_text = "Blisko" if vehicle_data['is_close'] else "Daleko"
                cv2.putText(display_frame, proximity_text, (x, y + h + 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if vehicle_data['captures'] > 0:
                    capture_text = f"Zapisano"
                    cv2.putText(display_frame, capture_text, (x, y + h + 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.putText(display_frame, f"Pojazdy: {len(vehicle_tracking)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display_frame, f"Klatka: {frame_count}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        bottom_line_y = int(frame_height * bottom_region_threshold)
        cv2.line(display_frame, (0, bottom_line_y), (frame_width, bottom_line_y), (255, 0, 0), 2)
        cv2.putText(display_frame, "Strefa bliskości", (10, bottom_line_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.imshow('VehicleStopDetector', display_frame)
        cv2.imshow('Maska', fgMask)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

video_path = '/Users/piotrwaluszek/Downloads/test2.mp4'
detect_stopped_vehicles(video_path)
