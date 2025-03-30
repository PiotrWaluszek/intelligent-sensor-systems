import os
import sys
import requests
import base64
import time
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
import io
import cv2
import numpy as np

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

MAX_PROCESSING_WIDTH = 800
MAX_PROCESSING_HEIGHT = 600

def check_image_file(file_path):
    start_time = time.time()
    
    if not os.path.exists(file_path):
        print(f"Błąd: Plik {file_path} nie istnieje.")
        return False
    
    try:
        img = Image.open(file_path)
        img.verify()
        
        with Image.open(file_path) as img:
            width, height = img.size
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        elapsed_time = time.time() - start_time
        print(f"Weryfikacja pliku: {elapsed_time:.3f} s")
        print(f"Oryginalny rozmiar obrazu: {width}x{height} pikseli, {file_size_mb:.2f} MB")
        
        return True
    except Exception as e:
        print(f"Błąd: {file_path} nie jest poprawnym plikiem obrazu. Szczegóły: {e}")
        return False

def resize_image_if_needed(file_path, max_size_mb=10, max_width=MAX_PROCESSING_WIDTH, max_height=MAX_PROCESSING_HEIGHT):
    start_time = time.time()
    
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    with Image.open(file_path) as img:
        width, height = img.size
    
    needs_resize_for_size = file_size_mb > max_size_mb
    needs_resize_for_dimensions = width > max_width or height > max_height
    
    if not (needs_resize_for_size or needs_resize_for_dimensions):
        elapsed_time = time.time() - start_time
        print(f"Sprawdzenie rozmiaru obrazu: {elapsed_time:.3f} s - obraz nie wymaga przeskalowania")
        return file_path
    
    img = Image.open(file_path)
    
    scale_factor_width = max_width / width if width > max_width else 1
    scale_factor_height = max_height / height if height > max_height else 1
    scale_factor_size = (max_size_mb / file_size_mb) ** 0.5 if needs_resize_for_size else 1
    
    scale_factor = min(scale_factor_width, scale_factor_height, scale_factor_size)
    
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    print(f"Przeskalowanie obrazu: {width}x{height} -> {new_width}x{new_height} (współczynnik: {scale_factor:.3f})")
    
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    
    temp_path = f"temp_{os.path.basename(file_path)}"
    resized_img.save(temp_path, quality=85, optimize=True)
    
    new_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    elapsed_time = time.time() - start_time
    
    print(f"Obraz zmniejszony z {file_size_mb:.2f} MB do {new_size_mb:.2f} MB w {elapsed_time:.3f} s")
    print(f"Nowy rozmiar obrazu: {new_width}x{new_height} pikseli")
    
    return temp_path

def encode_image(image_path):
    start_time = time.time()
    
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8')
    
    elapsed_time = time.time() - start_time
    print(f"Kodowanie obrazu do base64: {elapsed_time:.3f} s")
    
    return encoded

def recognize_license_plate(image_path):
    total_start_time = time.time()
    
    if not API_KEY:
        print("Błąd: Brak klucza API OpenAI. Sprawdź plik .env")
        return None
    
    processed_image_path = resize_image_if_needed(
        image_path, 
        max_size_mb=4,
        max_width=MAX_PROCESSING_WIDTH, 
        max_height=MAX_PROCESSING_HEIGHT
    )
    
    base64_image = encode_image(processed_image_path)
    
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
        print("Wysyłanie zapytania do OpenAI API...")
        api_start_time = time.time()
        
        response = requests.post("https://api.openai.com/v1/chat/completions", 
                                headers=headers, 
                                json=payload)
        
        api_elapsed_time = time.time() - api_start_time
        print(f"Czas odpowiedzi API: {api_elapsed_time:.3f} s")
        
        if response.status_code == 200:
            result = response.json()
            plate_number = result["choices"][0]["message"]["content"].strip()
            
            total_elapsed_time = time.time() - total_start_time
            print(f"Całkowity czas rozpoznawania: {total_elapsed_time:.3f} s")
            
            return plate_number
        else:
            print(f"Błąd API: {response.status_code}")
            print(f"Szczegóły: {response.text}")
            
            total_elapsed_time = time.time() - total_start_time
            print(f"Całkowity czas (z błędem): {total_elapsed_time:.3f} s")
            
            return None
    except Exception as e:
        print(f"Wystąpił błąd: {e}")
        
        total_elapsed_time = time.time() - total_start_time
        print(f"Całkowity czas (z wyjątkiem): {total_elapsed_time:.3f} s")
        
        return None

def preprocess_for_plate_detection(image_path):
    start_time = time.time()
    
    img = cv2.imread(image_path)
    if img is None:
        print("Nie można wczytać obrazu do przetwarzania.")
        return None
    
    original_height, original_width = img.shape[:2]
    
    if original_width > MAX_PROCESSING_WIDTH or original_height > MAX_PROCESSING_HEIGHT:
        scale_factor = min(MAX_PROCESSING_WIDTH / original_width, MAX_PROCESSING_HEIGHT / original_height)
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        print(f"Obraz przeskalowany dla przetwarzania: {original_width}x{original_height} -> {new_width}x{new_height}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(blur, 50, 150)
    
    temp_path = f"temp_processed_{os.path.basename(image_path)}"
    cv2.imwrite(temp_path, edges)
    
    elapsed_time = time.time() - start_time
    print(f"Wstępne przetwarzanie obrazu: {elapsed_time:.3f} s")
    
    return temp_path

def main():
    overall_start_time = time.time()
    
    print(f"=== Rozpoczęcie programu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    if len(sys.argv) != 2:
        print("Użycie: python license_plate_reader.py <ścieżka_do_obrazu>")
        return
    
    image_path = sys.argv[1]
    
    if not check_image_file(image_path):
        return
    
    plate_number = recognize_license_plate(image_path)
    
    if plate_number:
        if plate_number == "NIEROZPOZNANA":
            print("Nie udało się rozpoznać tablicy rejestracyjnej.")
        else:
            print(f"Rozpoznany numer tablicy rejestracyjnej: {plate_number}")
    else:
        print("Wystąpił błąd podczas rozpoznawania tablicy rejestracyjnej.")
    
    overall_elapsed_time = time.time() - overall_start_time
    print(f"=== Całkowity czas wykonania programu: {overall_elapsed_time:.3f} s ===")

if __name__ == "__main__":
    main()
