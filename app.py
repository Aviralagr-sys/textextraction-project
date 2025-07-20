from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import json
import tempfile
import shutil
from dotenv import load_dotenv
from flask_cors import CORS
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# Set Tesseract path explicitly
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, origins=["http://localhost:8501", "http://127.0.0.1:8501"],
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"])

# Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SUPPORTED_FILE_TYPES = [".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
MIN_TEXT_LENGTH = 50  # Minimum characters to consider a page has readable text

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


def allowed_file(filename):
    return '.' in filename and os.path.splitext(filename)[1].lower() in SUPPORTED_FILE_TYPES


def extract_text_with_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num in range(len(doc)):
        try:
            page = doc.load_page(page_num)
            text = page.get_text("text") or ""
            if len(text.strip()) < MIN_TEXT_LENGTH:
                try:
                    pix = page.get_pixmap(dpi=200)
                    img = Image.open(io.BytesIO(pix.tobytes("ppm")))
                    ocr_config = '--oem 1 --psm 6'
                    text = pytesseract.image_to_string(img, lang='eng', config=ocr_config)
                    text = f"[OCR EXTRACTED]\n{text}\n"
                except Exception as ocr_error:
                    print(f"OCR failed on page {page_num}: {str(ocr_error)}")
                    text = ""
            full_text += text + "\n\n"
        except Exception as page_error:
            print(f"Error processing page {page_num}: {page_error}")
            continue
    doc.close()
    return full_text.strip()


def extract_text_from_image_file(file_obj):
    img = Image.open(file_obj)
    config = '--oem 1 --psm 6'
    text = pytesseract.image_to_string(img, lang='eng', config=config)
    return text


def highlight_pdf_pages(pdf_path, keywords, output_dir="./highlighted_pages"):
    """
    Highlights keywords in the PDF and draws a red bounding box.
    Returns a list of image file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
    saved_images = []

    for page_num, page in enumerate(doc, start=1):
        found = False
        for keyword in keyword_list:
            matches = page.search_for(keyword)
            for rect in matches:
                # Yellow highlight
                page.add_highlight_annot(rect)
                # Red bounding box (rubber band)
                rubber_band = page.add_rect_annot(rect)
                rubber_band.set_colors(stroke=(1, 0, 0))
                rubber_band.set_border(width=2)
                rubber_band.update()
                found = True

        if found:
            pix = page.get_pixmap(dpi=200)
            image_path = os.path.join(output_dir, f"page_{page_num}.png")
            pix.save(image_path)
            saved_images.append(image_path)

    doc.close()
    return saved_images


@app.route('/highlighted_pages/<path:filename>')
def serve_highlighted_image(filename):
    return send_from_directory('./highlighted_pages', filename)


@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    if 'keywords' not in request.form:
        return jsonify({"error": "No keywords provided"}), 400

    files = request.files.getlist('files')
    keywords = request.form['keywords']

    if not files or all(file.filename == '' for file in files):
        return jsonify({"error": "No selected files"}), 400

    documents = []
    temp_dir = tempfile.mkdtemp()
    highlighted_pages = []

    try:
        for file in files:
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(temp_dir, filename)
                extension = os.path.splitext(filename)[1].lower()

                # PDF section
                if extension == '.pdf':
                    file.save(filepath)
                    try:
                        loader = PyPDFLoader(filepath)
                        docs = loader.load()
                        if sum(len(d.page_content) for d in docs) < MIN_TEXT_LENGTH * len(docs):
                            full_text = extract_text_with_ocr(filepath)
                            if full_text.strip():
                                docs = [Document(page_content=full_text)]
                    except Exception:
                        full_text = extract_text_with_ocr(filepath)
                        docs = [Document(page_content=full_text)] if full_text.strip() else []
                    documents.extend(docs)

                    # Highlighting
                    highlighted_images = highlight_pdf_pages(filepath, keywords)
                    highlighted_pages.extend([os.path.basename(img) for img in highlighted_images])

                elif extension == '.docx':
                    file.save(filepath)
                    loader = Docx2txtLoader(filepath)
                    documents.extend(loader.load())
                elif extension == '.txt':
                    file.save(filepath)
                    loader = TextLoader(filepath)
                    documents.extend(loader.load())
                elif extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                    try:
                        text = extract_text_from_image_file(file)
                        if text.strip():
                            documents.append(Document(page_content=f"[OCR IMAGE]\n{text.strip()}"))
                    except Exception as err:
                        print(f"OCR failed for image {filename}: {err}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    if not documents:
        return jsonify({"error": "No valid content extracted"}), 400

    # Chunk/split and analyze
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=400,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    filtered_splits = []
    for doc in splits:
        content = doc.page_content.strip()
        if (len(content) > 100 and
            not content.count('_') > len(content) * 0.3 and
            not content.count('.') > len(content) * 0.1):
            filtered_splits.append(doc)

    full_text = "\n\n".join([doc.page_content for doc in filtered_splits])

    prompt_template = """You are an expert document analyzer. Extract comprehensive and meaningful information from the following text based on these keywords: {keywords}

    Return the information in this JSON format:
    {{
      "results": [
        {{
          "keyword": "keyword_name",
          "definition": "What this keyword refers to in the document",
          "key_details": "Important facts and specifications",
          "programs_courses": "Specific programs, courses, or offerings",
          "requirements": "Prerequisites, requirements, or conditions",
          "additional_context": "Other relevant information",
          "relevant_quotes": "Direct quotes from the document"
        }}
      ]
    }}
    Return ONLY the JSON structure, nothing else."""

    prompt = PromptTemplate.from_template(prompt_template)

    chain = (
        {"keywords": RunnablePassthrough(), "text": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    try:
        result = chain.invoke({
            "keywords": keywords,
            "text": full_text[:15000]
        })
        start_idx = result.find('{')
        end_idx = result.rfind('}') + 1

        if start_idx == -1 or end_idx == 0:
            return jsonify({"status": "success", "data": {"results": [{"keyword": keywords, "relevant_text": result}]}})

        json_result = result[start_idx:end_idx]
        try:
            parsed_json = json.loads(json_result)
            return jsonify({
                "status": "success",
                "data": parsed_json,
                "highlighted_pages": highlighted_pages
            })
        except json.JSONDecodeError:
            return jsonify({
                "status": "success",
                "data": {"results": [{"keyword": keywords, "relevant_text": result}]},
                "highlighted_pages": highlighted_pages
            })

    except Exception as e:
        return jsonify({"error": "Failed to process keywords", "details": str(e)}), 500


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Document Information Extractor API",
        "endpoints": {
            "/health": "GET - Health check",
            "/upload": "POST - Upload files and extract information"
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "Service is running"})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
