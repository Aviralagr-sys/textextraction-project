import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import docx
import os
import re

from transformers import pipeline
from googletrans import Translator
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load models
print("🔁 Loading models...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
translator = Translator()
kw_model = KeyBERT(model=SentenceTransformer("all-MiniLM-L6-v2"))
print("✅ Models loaded successfully.")

# ---------- Extractors ----------
def extract_from_image(file_path):
    image = Image.open(file_path)
    return pytesseract.image_to_string(image)

def extract_from_pdf(file_path):
    doc = fitz.open(file_path)
    return "\n".join([page.get_text() for page in doc])

def extract_from_word(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

# ---------- Formatter ----------
def format_text(text):
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'•|·|●', '-', text)
    text = re.sub(r'-\n', '', text)
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(lines).strip()

# ---------- Summarizer (chunked) ----------
def summarize_text(text):
    try:
        if len(text.split()) < 50:
            return "Text too short to summarize."

        chunks = []
        words = text.split()
        chunk_size = 300

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            summary = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
            chunks.append(summary[0]['summary_text'])

        return "\n".join(chunks)
    except Exception as e:
        return f"Summarization failed: {e}"

# ---------- Translator (chunked) ----------
def translate_text(text, dest_language="hi"):
    try:
        if not text:
            return "Nothing to translate."

        chunks = [text[i:i + 4500] for i in range(0, len(text), 4500)]
        translated = [translator.translate(chunk, dest=dest_language).text for chunk in chunks]
        return "\n".join(translated)
    except Exception as e:
        return f"Translation failed: {e}"

# ---------- Keyword Extractor ----------
def extract_keywords(text, top_n=10):
    try:
        keywords = kw_model.extract_keywords(text, top_n=top_n)
        return [kw[0] for kw in keywords]
    except Exception as e:
        return [f"Keyword extraction failed: {e}"]

# ---------- Save Result ----------
def save_text_to_folder(original_file_path, formatted_text, include_summary=True, include_translation=True, include_keywords=True, output_folder="extracted_texts"):
    os.makedirs(output_folder, exist_ok=True)
    base_name = os.path.basename(original_file_path)
    file_name_without_ext = os.path.splitext(base_name)[0]
    output_path = os.path.join(output_folder, f"{file_name_without_ext}.txt")

    summary = summarize_text(formatted_text) if include_summary else None
    translation = translate_text(formatted_text) if include_translation else None
    keywords = extract_keywords(formatted_text) if include_keywords else None

    with open(output_path, "w", encoding="utf-8") as f:
        if not (include_summary or include_translation or include_keywords):
            f.write("------ Formatted Text ------\n")
            f.write(formatted_text + "\n")
        else:
            if include_summary:
                f.write("------ Summary ------\n")
                f.write(summary + "\n\n")
            if include_translation:
                f.write("------ Translated to Hindi ------\n")
                f.write(translation + "\n\n")
            if include_keywords:
                f.write("------ Keywords ------\n")
                f.write(", ".join(keywords) + "\n")

    print(f"\n✅ Results saved to: {output_path}")

# ---------- Main ----------
def main():
    file_path = input("Enter file path (or press Enter to use default): ").strip().strip('"')
    if not file_path:
        file_path = r"C:\Users\aviral\Downloads\intern_task_breakdown.pdf"

    if not os.path.exists(file_path):
        print("❌ File not found!")
        return

    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in [".png", ".jpg", ".jpeg", ".bmp"]:
            extracted_text = extract_from_image(file_path)
        elif ext == ".pdf":
            extracted_text = extract_from_pdf(file_path)
        elif ext == ".docx":
            extracted_text = extract_from_word(file_path)
        else:
            print("❌ Unsupported file type!")
            return

        formatted_text = format_text(extracted_text)

        print("\nWhat do you want to extract?")
        print("1. Full formatted text only")
        print("2. Summary and keywords only")
        print("3. All of the above")
        choice = input("Enter choice (1/2/3): ").strip()

        if choice == "1":
            save_text_to_folder(file_path, formatted_text, include_summary=False, include_translation=False, include_keywords=False)
        elif choice == "2":
            save_text_to_folder(file_path, formatted_text, include_summary=True, include_translation=False, include_keywords=True)
        elif choice == "3":
            save_text_to_folder(file_path, formatted_text, include_summary=True, include_translation=True, include_keywords=True)
        else:
            print("❌ Invalid choice!")

    except Exception as e:
        print(f"❌ An error occurred: {e}")

# ---------- Run ----------
if __name__ == "__main__":
    main()
