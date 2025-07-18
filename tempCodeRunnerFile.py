from flask import Flask, request, jsonify, abort
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

import os
import json  # Add this import
from dotenv import load_dotenv
import tempfile
import shutil
from flask_cors import CORS

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, origins=["http://localhost:8501", "http://127.0.0.1:8501"], 
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"])

# Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SUPPORTED_FILE_TYPES = [".pdf", ".docx", ".txt"]

# Initialize LLM
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'pdf', 'docx', 'txt'}

@app.errorhandler(HTTPException) 
def handle_exception(e):
    return jsonify({
        "error": e.name,
        "message": e.description
    }), e.code

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads and process them with keywords"""
    print("=== UPLOAD REQUEST RECEIVED ===")
    print(f"Request files: {request.files}")
    print(f"Request form: {request.form}")
    
    if 'files' not in request.files:
        print("ERROR: No files in request")
        return jsonify({"error": "No files uploaded"}), 400
        
    if 'keywords' not in request.form:
        print("ERROR: No keywords in request")
        return jsonify({"error": "No keywords provided"}), 400
        
    files = request.files.getlist('files')
    keywords = request.form['keywords']
    
    print(f"Files received: {len(files)}")
    print(f"Keywords: {keywords}")
    
    if not files or all(file.filename == '' for file in files):
        print("ERROR: No valid files selected")
        return jsonify({"error": "No selected files"}), 400
    
    if not keywords:
        print("ERROR: Keywords empty")
        return jsonify({"error": "Keywords cannot be empty"}), 400
    
    documents = []
    temp_dir = tempfile.mkdtemp()
    print(f"Created temp directory: {temp_dir}")
    
    try:
        for file in files:
            print(f"Processing file: {file.filename}")
            try:
                if file and file.filename and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(temp_dir, filename)
                    file.save(filepath)
                    print(f"Saved file to: {filepath}")
                    
                    # Initialize loader as None
                    loader = None
                    
                    if filename.lower().endswith('.pdf'):
                        loader = PyPDFLoader(filepath)
                    elif filename.lower().endswith('.docx'):
                        loader = Docx2txtLoader(filepath)
                    elif filename.lower().endswith('.txt'):
                        loader = TextLoader(filepath)
                    
                    # Only load if we have a valid loader
                    if loader is not None:
                        docs = loader.load()
                        documents.extend(docs)
                        print(f"Loaded {len(docs)} documents from {filename}")
                    else:
                        print(f"WARNING: Unsupported file format: {filename}")
                        
            except Exception as e:
                print(f"ERROR processing {file.filename}: {str(e)}")
                continue  # Continue with other files
    
    finally:
        # Clean up temp files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temp directory: {temp_dir}")
    
    print(f"Total documents loaded: {len(documents)}")
    if not documents:
        print("ERROR: No valid content extracted")
        return jsonify({"error": "No valid content extracted from uploaded files"}), 400
    
    # Split documents into chunks with better overlap for context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # Increased chunk size for better context
        chunk_overlap=400,  # Increased overlap
        separators=["\n\n", "\n", ". ", " ", ""]  # Better separators
    )
    splits = text_splitter.split_documents(documents)
    
    # Filter out very short chunks and table of contents
    filtered_splits = []
    for doc in splits:
        content = doc.page_content.strip()
        # Filter out table of contents patterns and very short content
        if (len(content) > 100 and 
            not content.count('_') > len(content) * 0.3 and  # Skip lines with too many underscores
            not content.count('.') > len(content) * 0.1):    # Skip lines with too many dots
            filtered_splits.append(doc)
    
    # Combine filtered text content
    full_text = "\n\n".join([doc.page_content for doc in filtered_splits])
    
    print(f"Total text length after filtering: {len(full_text)} characters")
    print(f"First 500 characters: {full_text[:500]}")  # Debug preview
    
    # Create an improved prompt to extract more meaningful information
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
        # Also try to extract keyword-specific content
        keyword_list = [k.strip() for k in keywords.split(',')]
        
        # For each keyword, find the most relevant chunks
        keyword_specific_content = {}
        for keyword in keyword_list:
            relevant_chunks = []
            for doc in filtered_splits:
                content = doc.page_content.lower()
                if keyword.lower() in content:
                    # Get surrounding context
                    relevant_chunks.append(doc.page_content)
            
            if relevant_chunks:
                # Take the most relevant chunks (up to 3 per keyword)
                keyword_specific_content[keyword] = "\n\n".join(relevant_chunks[:3])
        
        # If we found keyword-specific content, use it; otherwise use full text
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
            "text": analysis_text[:15000]  # Limit to prevent token overflow
        })
        
        # Clean up the response to ensure it's valid JSON
        # Sometimes LLMs add extra text before/after the JSON
        start_idx = result.find('{')
        end_idx = result.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            # If no JSON found, return a structured response
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
        
        # Parse the JSON to ensure it's valid
        try:
            parsed_json = json.loads(json_result)
            return jsonify({
                "status": "success",
                "data": parsed_json
            })
        except json.JSONDecodeError as e:
            app.logger.error(f"JSON parsing error: {str(e)}")
            # Return a fallback response
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
    """Home endpoint"""
    return jsonify({
        "message": "Document Information Extractor API",
        "endpoints": {
            "/health": "GET - Health check",
            "/upload": "POST - Upload files and extract information"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "message": "Service is running"
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)