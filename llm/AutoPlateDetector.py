import cv2
import numpy as np
import time
import os
import sys
import requests
import base64
import logging
import shutil
import threading
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
MAX_PROCESSING_WIDTH = 800
MAX_PROCESSING_HEIGHT = 600

class PlateReaderConnector:
    def __init__(self, 
                 input_folder='detected_vehicles',
                 processed_folder='processed_vehicles',
                 failed_folder='failed_vehicles',
                 log_file='plate_reader.log',
                 check_interval=5):
        for folder in [input_folder, processed_folder, failed_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
        
        self.input_folder = input_folder
        self.processed_folder = processed_folder
        self.failed_folder = failed_folder
        self.check_interval = check_interval
        
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('PlateReaderConnector')
        
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        self.logger.addHandler(console)
        
        self.logger.info("PlateReaderConnector zainicjalizowany")
        self.running = True
    
    def check_image_file(self, file_path):
        start_time = time.time()
        
        if not os.path.exists(file_path):
            self.logger.error(f"Błąd: Plik {file_path} nie istnieje.")
            return False
        
        try:
            img = Image.open(file_path)
            img.verify()
            
            with Image.open(file_path) as img:
                width, height = img.size
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Weryfikacja pliku: {elapsed_time:.3f} s")
            self.logger.info(f"Oryginalny rozmiar obrazu: {width}x{height} pikseli, {file_size_mb:.2f} MB")
            
            return True
        except Exception as e:
            self.logger.error(f"Błąd: {file_path} nie jest poprawnym plikiem obrazu. Szczegóły: {e}")
            return False
    
    def resize_image_if_needed(self, file_path, max_size_mb=10, max_width=MAX_PROCESSING_WIDTH, max_height=MAX_PROCESSING_HEIGHT):
        start_time = time.time()
        
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        with Image.open(file_path) as img:
            width, height = img.size
        
        needs_resize_for_size = file_size_mb > max_size_mb
        needs_resize_for_dimensions = width > max_width or height > max_height
        
        if not (needs_resize_for_size or needs_resize_for_dimensions):
            elapsed_time = time.time() - start_time
            self.logger.info(f"Sprawdzenie rozmiaru obrazu: {elapsed_time:.3f} s - obraz nie wymaga przeskalowania")
            return file_path
        
        img = Image.open(file_path)
        
        scale_factor_width = max_width / width if width > max_width else 1
        scale_factor_height = max_height / height if height > max_height else 1
        scale_factor_size = (max_size_mb / file_size_mb) ** 0.5 if needs_resize_for_size else 1
        
        scale_factor = min(scale_factor_width, scale_factor_height, scale_factor_size)
        
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        self.logger.info(f"Przeskalowanie obrazu: {width}x{height} -> {new_width}x{new_height} (współczynnik: {scale_factor:.3f})")
        
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        
        temp_path = f"temp_{os.path.basename(file_path)}"
        resized_img.save(temp_path, quality=85, optimize=True)
        
        new_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        elapsed_time = time.time() - start_time
        
        self.logger.info(f"Obraz zmniejszony z {file_size_mb:.2f} MB do {new_size_mb:.2f} MB w {elapsed_time:.3f} s")
        self.logger.info(f"Nowy rozmiar obrazu: {new_width}x{new_height} pikseli")
        
        return temp_path
    
    def encode_image(self, image_path):
        start_time = time.time()
        
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Kodowanie obrazu do base64: {elapsed_time:.3f} s")
        
        return encoded
    
    def recognize_license_plate(self, image_path):
        total_start_time = time.time()
        
        processed_image_path = self.resize_image_if_needed(
            image_path, 
            max_size_mb=4,
            max_width=MAX_PROCESSING_WIDTH, 
            max_height=MAX_PROCESSING_HEIGHT
        )
        
        base64_image = self.encode_image(processed_image_path)
        
        if processed_image_path != image_path:
            os.remove(processed_image_path)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Na tym zdjęciu znajduje się samochód z tablicą rejestracyjną. Rozpoznaj i podaj TYLKO numer tablicy rejestracyjnej, bez żadnych dodatkowych informacji czy wyjaśnień. Jeśli nie możesz rozpoznać tablicy, zwróć tylko tekst 'NIEROZPOZNANA'."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 50
        }
        
        try:
            self.logger.info("Wysyłanie zapytania do OpenAI API...")
            api_start_time = time.time()
            
            response = requests.post("https://api.openai.com/v1/chat/completions", 
                                    headers=headers, 
                                    json=payload)
            
            api_elapsed_time = time.time() - api_start_time
            self.logger.info(f"Czas odpowiedzi API: {api_elapsed_time:.3f} s")
            
            if response.status_code == 200:
                result = response.json()
                plate_number = result["choices"][0]["message"]["content"].strip()
                
                total_elapsed_time = time.time() - total_start_time
                self.logger.info(f"Całkowity czas rozpoznawania: {total_elapsed_time:.3f} s")
                
                return plate_number
            else:
                self.logger.error(f"Błąd API: {response.status_code}")
                self.logger.error(f"Szczegóły: {response.text}")
                
                total_elapsed_time = time.time() - total_start_time
                self.logger.info(f"Całkowity czas (z błędem): {total_elapsed_time:.3f} s")
                
                return None
        except Exception as e:
            self.logger.error(f"Wystąpił błąd: {e}")
            
            total_elapsed_time = time.time() - total_start_time
            self.logger.info(f"Całkowity czas (z wyjątkiem): {total_elapsed_time:.3f} s")
            
            return None
    
    def process_image(self, image_path):
        image_name = os.path.basename(image_path)
        self.logger.info(f"Przetwarzanie obrazu: {image_name}")
        
        if not self.check_image_file(image_path):
            destination = os.path.join(self.failed_folder, image_name)
            shutil.move(image_path, destination)
            self.logger.warning(f"Obraz {image_name} przeniesiony do {self.failed_folder} (niepoprawny format)")
            return
        
        plate_number = self.recognize_license_plate(image_path)
        
        result_file = os.path.join(self.processed_folder, f"{os.path.splitext(image_name)[0]}_result.txt")
        
        if plate_number:
            with open(result_file, 'w') as f:
                f.write(f"Obraz: {image_name}\n")
                f.write(f"Rozpoznana tablica: {plate_number}\n")
                f.write(f"Czas rozpoznania: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            destination = os.path.join(self.processed_folder, image_name)
            shutil.move(image_path, destination)
            self.logger.info(f"Obraz {image_name} przeniesiony do {self.processed_folder}")
            self.logger.info(f"Rozpoznana tablica: {plate_number}")
        else:
            with open(result_file, 'w') as f:
                f.write(f"Obraz: {image_name}\n")
                f.write(f"Błąd rozpoznawania tablicy\n")
                f.write(f"Czas próby: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            destination = os.path.join(self.failed_folder, image_name)
            shutil.move(image_path, destination)
            self.logger.warning(f"Obraz {image_name} przeniesiony do {self.failed_folder} (błąd rozpoznawania)")
    
    def run(self):
        self.logger.info("Rozpoczęcie monitorowania folderu z obrazami")
        
        try:
            while self.running:
                image_files = [f for f in os.listdir(self.input_folder) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if image_files:
                    self.logger.info(f"Znaleziono {len(image_files)} nowych obrazów do przetworzenia")
                    
                    for image_file in image_files:
                        if not self.running:
                            break
                        image_path = os.path.join(self.input_folder, image_file)
                        self.process_image(image_path)
                
                time.sleep(self.check_interval)
                
        except Exception as e:
            self.logger.error(f"Wystąpił nieoczekiwany błąd: {str(e)}")
    
    def stop(self):
        self.running = False
        self.logger.info("Zatrzymywanie PlateReaderConnector")

def detect_stopped_vehicles(video_path, output_folder='detected_vehicles', connector=None):
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
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if connector:
        connector.stop()

def main():
    if len(sys.argv) != 2:
        print("Użycie: python AutoPlateDetector.py <ścieżka_do_pliku_wideo>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_folder = 'detected_vehicles'
    
    if not API_KEY:
        print("Błąd: Brak klucza API OpenAI. Sprawdź plik .env")
        sys.exit(1)
    
    connector = PlateReaderConnector(
        input_folder=output_folder,
        processed_folder='processed_vehicles',
        failed_folder='failed_vehicles',
        check_interval=5
    )
    
    connector_thread = threading.Thread(target=connector.run)
    connector_thread.daemon = True
    connector_thread.start()
    
    try:
        detect_stopped_vehicles(video_path, output_folder, connector)
    except KeyboardInterrupt:
        print("Program zatrzymany przez użytkownika")
    except Exception as e:
        print(f"Wystąpił błąd: {e}")
    finally:
        connector.stop()
        connector_thread.join(timeout=5)

if __name__ == "__main__":
    main()
