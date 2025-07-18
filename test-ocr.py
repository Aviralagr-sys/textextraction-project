from PIL import Image
import pytesseract

# If Tesseract is not on your PATH, uncomment and set manually:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image_path = r'C:\internship\textextractor\test-image.png'

try:
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    print("===== OCR Output =====")
    print(text)
except Exception as e:
    print("Error during OCR:", e)
