import cv2
import numpy as np
import easyocr
import time
import os
import re
import platform
import threading
from ultralytics import YOLO
from datetime import datetime
import queue

is_macos = platform.system() == 'Darwin'
POLISH_PLATE_PATTERN = r'^[A-Z]{1,3}[\s\-]?[A-Z0-9]{4,5}$'

frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)
detected_text_queue = queue.Queue(maxsize=1)

processing_active = True
fps_counter = 0
fps_timer = time.time()
current_fps = 0

PROCESS_EVERY_N_FRAMES = 3

PROCESSING_WIDTH = 800
PROCESSING_HEIGHT = 600

ENABLE_OCR = True

DISPLAY_DETECTION_TIME = 10.0

MIN_PLATE_TEXT_LENGTH = 7

def init_ocr():
    if not ENABLE_OCR:
        return None
    
    print("Inicjalizacja EasyOCR...")
    return easyocr.Reader(['en', 'pl'], gpu=False)

def init_model():
    print("Ładowanie modelu YOLO...")
    try:
        model_path = '/Users/piotrwaluszek/iss/yolo11n.pt'
        
        if os.path.exists(model_path):
            model = YOLO(model_path)
            print(f"Załadowano YOLO z lokalnego pliku: {model_path}")
            return model
        else:
            print(f"Nie znaleziono pliku modelu: {model_path}")
            model = YOLO('yolov8n.pt')
            print("Załadowano YOLOv8n")
            return model
    except Exception as e:
        print(f"Wystąpił błąd podczas inicjalizacji modelu: {e}")
        model = YOLO('yolov8n.pt')
        print("Załadowano YOLOv8n")
        return model

def save_license_plate_to_file(license_plate_text, source_info):
    output_dir = "detected_plates"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"detected_plates_{timestamp}.txt")
    
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"Data i czas: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Źródło: {source_info}\n")
        f.write(f"Wykryty numer tablicy: {license_plate_text}\n")
        f.write("-" * 50 + "\n")
    
    print(f"Zapisano numer tablicy do pliku: {output_file}")
    
    all_plates_file = os.path.join(output_dir, "all_detected_plates.txt")
    with open(all_plates_file, 'a', encoding='utf-8') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {source_info} | {license_plate_text}\n")

def enhance_plate_image(plate_img):
    if plate_img is None or plate_img.size == 0:
        return None
    
    plate_img = cv2.resize(plate_img, (plate_img.shape[1]*2, plate_img.shape[0]*2), interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    blur = cv2.bilateralFilter(enhanced, 11, 17, 17)
    
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    inverted = cv2.bitwise_not(thresh)
    
    return [thresh, inverted, enhanced]

def detect_license_plates_with_yolo(image, model):
    h, w = image.shape[:2]
    scale = min(PROCESSING_WIDTH / w, PROCESSING_HEIGHT / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    if new_w < w or new_h < h:
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized_image = image
    
    results = model(resized_image, conf=0.25, verbose=False)
    
    license_plates = []
    boxes = []
    
    scale_x = w / new_w if new_w < w else 1
    scale_y = h / new_h if new_h < h else 1
    
    has_license_plate_class = False
    class_names = model.names
    license_plate_class_id = None
    
    for class_id, class_name in class_names.items():
        if 'license' in class_name.lower() or 'plate' in class_name.lower() or 'tablica' in class_name.lower():
            has_license_plate_class = True
            license_plate_class_id = class_id
            break
    
    if has_license_plate_class and license_plate_class_id is not None:
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls == license_plate_class_id:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
                    
                    plate_img = image[y1:y2, x1:x2]
                    if plate_img.size > 0:
                        license_plates.append(plate_img)
                        boxes.append((x1, y1, x2, y2))
    
    if not license_plates:
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls == 2:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
                    
                    car_region = image[y1:y2, x1:x2]
                    if car_region.size == 0:
                        continue
                    
                    hsv = cv2.cvtColor(car_region, cv2.COLOR_BGR2HSV)
                    lower_blue = np.array([100, 50, 50])
                    upper_blue = np.array([130, 255, 255])
                    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
                    
                    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in blue_contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        if w > 10 and h > 10:
                            plate_width = min(int(w * 7), car_region.shape[1] - x)
                            plate_height = min(int(h * 1.5), car_region.shape[0] - y)
                            
                            if x + plate_width <= car_region.shape[1] and y + plate_height <= car_region.shape[0]:
                                plate_img = car_region[y:y+plate_height, x:x+plate_width]
                                ratio = plate_width / plate_height if plate_height > 0 else 0
                                
                                if 2.0 < ratio < 7.0 and plate_width > 30 and plate_height > 10:
                                    license_plates.append(plate_img)
                                    global_px1 = x1 + x
                                    global_py1 = y1 + y
                                    global_px2 = x1 + x + plate_width
                                    global_py2 = y1 + y + plate_height
                                    boxes.append((global_px1, global_py1, global_px2, global_py2))
                    
                    if not license_plates:
                        lower_white = np.array([0, 0, 150])
                        upper_white = np.array([180, 30, 255])
                        white_mask = cv2.inRange(hsv, lower_white, upper_white)
                        
                        kernel = np.ones((5, 5), np.uint8)
                        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
                        
                        white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        for contour in white_contours:
                            x, y, w, h = cv2.boundingRect(contour)
                            ratio = float(w) / h if h > 0 else 0
                            
                            if 2.0 < ratio < 7.0 and w > 30 and h > 10:
                                roi = car_region[y:y+h, x:x+w]
                                if roi.size == 0:
                                    continue
                                
                                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                                avg_brightness = np.mean(gray_roi)
                                variance = np.var(gray_roi)
                                
                                if avg_brightness > 100 and variance > 500:
                                    license_plates.append(roi)
                                    global_px1 = x1 + x
                                    global_py1 = y1 + y
                                    global_px2 = x1 + x + w
                                    global_py2 = y1 + y + h
                                    boxes.append((global_px1, global_py1, global_px2, global_py2))
    
    return license_plates, boxes

def detect_license_plates(image, model):
    license_plates, boxes = detect_license_plates_with_yolo(image, model)
    
    if not license_plates:
        plates_info = []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        kernel = np.ones((5, 5), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        
        white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in white_contours:
            x, y, w, h = cv2.boundingRect(contour)
            ratio = float(w) / h
            
            if 2.0 < ratio < 7.0 and w > 60 and h > 20:
                roi = image[y:y+h, x:x+w]
                if roi.size == 0:
                    continue
                
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                avg_brightness = np.mean(gray_roi)
                variance = np.var(gray_roi)
                
                if avg_brightness > 100 and variance > 1000:
                    plates_info.append((roi, (x, y, x+w, y+h), 0.7))
        
        plates_info.sort(key=lambda x: x[2], reverse=True)
        
        license_plates = [info[0] for info in plates_info]
        boxes = [info[1] for info in plates_info]
    
    return license_plates, boxes

def correct_plate_text(text):
    corrections = {
        '0': 'O', '1': 'I', '8': 'B', '5': 'S', '2': 'Z',
        'D': '0', 'I': '1', 'B': '8', 'S': '5', 'Z': '2',
        'G': '6', '6': 'G', 'U': 'V', 'V': 'U'
    }
    
    if not text:
        return None
    
    text = text.upper().strip()
    
    if len(text) < 6 or len(text) > 9:
        return None
    
    text = re.sub(r'[^A-Z0-9]', '', text)
    
    if len(text) < 6:
        return None
    
    first_part = ""
    second_part = ""
    
    if len(text) == 6:
        first_part = text[0]
        second_part = text[1:]
    elif len(text) == 7:
        first_part = text[:2]
        second_part = text[2:]
    elif len(text) == 8:
        first_part = text[:3]
        second_part = text[3:]
    else:
        return None
    
    if not first_part.isalpha():
        for i, char in enumerate(first_part):
            if char.isdigit():
                first_part = first_part[:i] + corrections.get(char, char) + first_part[i+1:]
    
    for i, char in enumerate(second_part):
        if i < 2 and char.isalpha() and char in corrections:
            second_part = second_part[:i] + corrections.get(char, char) + second_part[i+1:]
    
    corrected_text = first_part + " " + second_part
    
    if re.match(POLISH_PLATE_PATTERN, corrected_text):
        return corrected_text
    
    return None

def recognize_license_plate(plate_img, reader):
    if not ENABLE_OCR or reader is None or plate_img is None or plate_img.size == 0:
        return None
    
    enhanced_variants = enhance_plate_image(plate_img)
    if enhanced_variants is None:
        return None
    
    variants = [plate_img, cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)] + enhanced_variants
    
    all_texts = []
    
    for variant in variants:
        try:
            results = reader.readtext(variant)
            
            for (bbox, text, prob) in results:
                cleaned_text = re.sub(r'[^\w\s\-]', '', text).strip().upper()
                
                if cleaned_text and prob > 0.2:
                    all_texts.append((cleaned_text, prob))
        except Exception as e:
            print(f"Błąd OCR: {e}")
            continue
    
    all_texts.sort(key=lambda x: x[1], reverse=True)
    
    for text, _ in all_texts:
        corrected = correct_plate_text(text)
        if corrected and len(re.sub(r'[^A-Z0-9]', '', corrected)) >= MIN_PLATE_TEXT_LENGTH:
            return corrected
    
    if all_texts:
        combined_text = "".join([t[0] for t in all_texts])
        corrected = correct_plate_text(combined_text)
        if corrected and len(re.sub(r'[^A-Z0-9]', '', corrected)) >= MIN_PLATE_TEXT_LENGTH:
            return corrected
    
    return None

def processing_thread(model, reader):
    global processing_active, fps_counter, fps_timer, current_fps
    
    frame_count = 0
    last_detected_text = None
    last_detection_frame = None
    last_detection_time = 0
    detection_active = True
    
    while processing_active:
        try:
            try:
                frame_to_process = frame_queue.get(block=False)
                frame_queue.task_done()
            except queue.Empty:
                time.sleep(0.001)
                continue
            
            frame_count += 1
            current_time = time.time()
            
            if not detection_active and current_time - last_detection_time >= DISPLAY_DETECTION_TIME:
                detection_active = True
                print("Wznowienie detekcji po okresie wyświetlania")
            
            if last_detection_frame is not None and not detection_active:
                frame_with_info = last_detection_frame.copy()
                
                cv2.putText(frame_with_info, f"FPS: {current_fps}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                remaining_time = int(DISPLAY_DETECTION_TIME - (current_time - last_detection_time))
                cv2.putText(frame_with_info, f"Nowa detekcja za: {remaining_time}s", (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                try:
                    result_queue.put(frame_with_info, block=False)
                except queue.Full:
                    pass
                
                continue
            
            if frame_count % PROCESS_EVERY_N_FRAMES != 0 or not detection_active:
                frame_with_info = frame_to_process.copy()
                
                cv2.putText(frame_with_info, f"FPS: {current_fps}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.putText(frame_with_info, "Nacisnij 'q', aby zakonczyc", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                try:
                    last_text = detected_text_queue.get(block=False)
                    detected_text_queue.task_done()
                    last_detected_text = last_text
                except queue.Empty:
                    pass
                
                if last_detected_text:
                    cv2.putText(frame_with_info, f"Ostatnia wykryta tablica: {last_detected_text}", 
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                try:
                    result_queue.put(frame_with_info, block=False)
                except queue.Full:
                    pass
                
                continue
            
            license_plates, boxes = detect_license_plates(frame_to_process, model)
            
            if license_plates:
                frame_with_plates = frame_to_process.copy()
                detected_valid_plate = False
                
                for i, (plate_img, box) in enumerate(zip(license_plates, boxes)):
                    x1, y1, x2, y2 = box
                    
                    if ENABLE_OCR and reader is not None:
                        text = recognize_license_plate(plate_img, reader)
                        
                        if text and len(re.sub(r'[^A-Z0-9]', '', text)) >= MIN_PLATE_TEXT_LENGTH:
                            detected_valid_plate = True
                            
                            cv2.rectangle(frame_with_plates, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            
                            if text != last_detected_text:
                                last_detected_text = text
                                try:
                                    detected_text_queue.put(text, block=False)
                                except queue.Full:
                                    pass
                                save_license_plate_to_file(text, "Kamera")
                            
                            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                            cv2.rectangle(frame_with_plates, (x1, y1-text_size[1]-10), (x1+text_size[0], y1), (0, 0, 0), -1)
                            cv2.putText(frame_with_plates, text, (x1, y1-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(frame_with_plates, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                cv2.putText(frame_with_plates, f"FPS: {current_fps}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.putText(frame_with_plates, "Nacisnij 'q', aby zakonczyc", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if last_detected_text:
                    cv2.putText(frame_with_plates, f"Wykryta tablica: {last_detected_text}", 
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if detected_valid_plate:
                    last_detection_frame = frame_with_plates
                    last_detection_time = current_time
                    detection_active = False
                    
                    print(f"Wykryto tablicę: {last_detected_text}")
                    print(f"Wyświetlanie przez {DISPLAY_DETECTION_TIME} sekund")
                
                try:
                    result_queue.put(frame_with_plates, block=False)
                except queue.Full:
                    pass
            else:
                frame_no_plate = frame_to_process.copy()
                
                cv2.putText(frame_no_plate, f"FPS: {current_fps}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.putText(frame_no_plate, "Nacisnij 'q', aby zakonczyc", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if last_detected_text:
                    cv2.putText(frame_no_plate, f"Ostatnia wykryta tablica: {last_detected_text}", 
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                try:
                    result_queue.put(frame_no_plate, block=False)
                except queue.Full:
                    pass
                
        except Exception as e:
            print(f"Błąd podczas przetwarzania: {e}")
            try:
                frame_error = frame_to_process.copy()
                cv2.putText(frame_error, "Błąd przetwarzania", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                result_queue.put(frame_error, block=False)
            except:
                pass

def process_from_camera(model, reader):
    global processing_active, fps_counter, fps_timer, current_fps
    
    camera_opened = False
    cap = None
    
    if is_macos:
        if not os.path.exists("Info.plist"):
            with open("Info.plist", "w") as f:
                f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                f.write('<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n')
                f.write('<plist version="1.0">\n')
                f.write('<dict>\n')
                f.write('    <key>NSCameraUseContinuityCameraDeviceType</key>\n')
                f.write('    <true/>\n')
                f.write('</dict>\n')
                f.write('</plist>\n')
            print("Utworzono plik Info.plist dla obsługi Continuity Camera")
        
        camera_configs = [
            (0, cv2.CAP_AVFOUNDATION),
            (1, cv2.CAP_AVFOUNDATION),
            (0, cv2.CAP_ANY),
            (1, cv2.CAP_ANY)
        ]
        
        for camera_index, api_preference in camera_configs:
            print(f"Próba otwarcia kamery z indeksem {camera_index} i API {api_preference}")
            cap = cv2.VideoCapture(camera_index, api_preference)
            
            time.sleep(0.5)
            
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None and test_frame.size > 0:
                    camera_opened = True
                    print(f"Kamera otwarta z indeksem {camera_index} i API {api_preference}")
                    break
                else:
                    cap.release()
    else:
        for camera_index in range(3):
            print(f"Próba otwarcia kamery z indeksem {camera_index}")
            cap = cv2.VideoCapture(camera_index)
            time.sleep(0.5)
            
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None and test_frame.size > 0:
                    camera_opened = True
                    print(f"Kamera otwarta z indeksem {camera_index}")
                    break
                else:
                    cap.release()
    
    if not camera_opened or cap is None:
        print("Nie można otworzyć kamery. Upewnij się, że:")
        print("1. Kamera jest podłączona i działa")
        print("2. Aplikacja ma uprawnienia do korzystania z kamery")
        print("3. Żadna inna aplikacja nie używa obecnie kamery")
        if is_macos:
            print("Na macOS sprawdź Preferencje systemowe > Bezpieczeństwo i prywatność > Prywatność > Kamera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Kamera uruchomiona. Naciśnij 'q', aby zakończyć.")
    print(f"Przetwarzanie co {PROCESS_EVERY_N_FRAMES} klatek")
    print(f"Czas wyświetlania wykrytej tablicy: {DISPLAY_DETECTION_TIME} sekund")
    print(f"Minimalna długość tekstu tablicy: {MIN_PLATE_TEXT_LENGTH} znaków")
    
    cv2.namedWindow('Detekcja tablicy rejestracyjnej', cv2.WINDOW_NORMAL)
    
    processing_active = True
    processor = threading.Thread(target=processing_thread, args=(model, reader))
    processor.daemon = True
    processor.start()
    
    running = True
    
    fps_counter = 0
    fps_timer = time.time()
    current_fps = 0
    
    try:
        while running:
            ret, frame = cap.read()
            
            if not ret or frame is None or frame.size == 0:
                print("Nie można odczytać klatki z kamery")
                time.sleep(0.01)
                continue
            
            fps_counter += 1
            if time.time() - fps_timer > 0.5:
                current_fps = int(fps_counter / (time.time() - fps_timer))
                fps_counter = 0
                fps_timer = time.time()
            
            try:
                frame_queue.put(frame.copy(), block=False)
            except queue.Full:
                pass
            
            try:
                processed_frame = result_queue.get(block=False)
                result_queue.task_done()
                cv2.imshow('Detekcja tablicy rejestracyjnej', processed_frame)
            except queue.Empty:
                preview_frame = frame.copy()
                cv2.putText(preview_frame, f"FPS: {current_fps}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('Detekcja tablicy rejestracyjnej', preview_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Klawisz 'q' naciśnięty - kończenie...")
                running = False
                break
    
    except KeyboardInterrupt:
        print("Program przerwany przez użytkownika")
    except Exception as e:
        print(f"Wystąpił błąd: {e}")
    finally:
        processing_active = False
        if processor.is_alive():
            processor.join(timeout=1.0)
        
        print("Zwalnianie zasobów kamery...")
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("Program zakończony")

def main():
    print("Program do detekcji polskich tablic rejestracyjnych")
    
    model = init_model()
    reader = init_ocr()
    
    process_from_camera(model, reader)

if __name__ == "__main__":
    print(f"System operacyjny: {platform.system()} {platform.release()}")
    print(f"Data i czas: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"Wersja OpenCV: {cv2.__version__}")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram przerwany przez użytkownika")
    except Exception as e:
        print(f"\nWystąpił nieoczekiwany błąd: {e}")
    finally:
        print("Zamykanie programu...")
        cv2.destroyAllWindows()
