from flask import Flask, request, jsonify
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

# Set Tesseract path explicitly for maximum reliability!
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
                        print(f"Received image file: {filename}")  # For debug
                        text = extract_text_from_image_file(file)
                        print(f"OCR output for {filename[:30]}: {text[:100]!r}")  # For debug
                        if text.strip():
                            documents.append(Document(page_content=f"[OCR IMAGE]\n{text.strip()}"))
                    except Exception as err:
                        print(f"OCR failed for image {filename}: {err}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    if not documents:
        print("DEBUG: No documents extracted from these files:", [file.filename for file in files])
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

    For each keyword, provide:
    1. **Definition/Description**: What the keyword refers to in this document
    2. **Key Details**: Important facts, requirements, or specifications
    3. **Programs/Courses**: Specific programs, courses, or offerings mentioned
    4. **Requirements**: Any prerequisites, requirements, or conditions
    5. **Additional Context**: Any other relevant information

    Keywords to analyze: {keywords}

    Document Text:
    {text}

    Instructions:
    - Look for substantial content, not just headers or table of contents
    - Extract specific details like course codes, credit hours, requirements, descriptions
    - Include relevant quotes and specific information
    - If a keyword appears in multiple contexts, include all relevant information
    - Focus on actionable and informative content
    - Provide detailed explanations, not just brief mentions

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
        keyword_list = [k.strip() for k in keywords.split(',')]
        keyword_specific_content = {}
        for keyword in keyword_list:
            relevant_chunks = []
            for doc in filtered_splits:
                content = doc.page_content.lower()
                if keyword.lower() in content:
                    relevant_chunks.append(doc.page_content)
            if relevant_chunks:
                keyword_specific_content[keyword] = "\n\n".join(relevant_chunks[:3])
        if keyword_specific_content:
            combined_content = "\n\n".join([
                f"=== Content for {keyword} ===\n{content}" 
                for keyword, content in keyword_specific_content.items()
            ])
            analysis_text = combined_content
        else:
            analysis_text = full_text
        result = chain.invoke({
            "keywords": keywords,
            "text": analysis_text[:15000]
        })
        start_idx = result.find('{')
        end_idx = result.rfind('}') + 1

        if start_idx == -1 or end_idx == 0:
            return jsonify({
                "status": "success",
                "data": {
                    "results": [{
                        "keyword": keywords,
                        "relevant_text": result,
                        "context": "Raw response from model"
                    }]
                }
            })

        json_result = result[start_idx:end_idx]
        try:
            parsed_json = json.loads(json_result)
            return jsonify({
                "status": "success",
                "data": parsed_json
            })
        except json.JSONDecodeError as e:
            app.logger.error(f"JSON parsing error: {str(e)}")
            return jsonify({
                "status": "success",
                "data": {
                    "results": [{
                        "keyword": keywords,
                        "relevant_text": result,
                        "context": "Could not parse as JSON, showing raw response"
                    }]
                }
            })

    except Exception as e:
        app.logger.error(f"Error processing keywords: {str(e)}")
        return jsonify({
            "error": "Failed to process keywords",
            "details": str(e)
        }), 500

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
    return jsonify({
        "status": "ok",
        "message": "Service is running"
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
