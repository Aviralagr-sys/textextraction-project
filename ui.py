import streamlit as st
import requests
import traceback

BACKEND_URL = "http://127.0.0.1:5001"

st.title("📄 Document & Image Information Extractor")
st.markdown("""
Upload PDFs, DOCX, TXT, or images (PNG, JPG, JPEG, BMP, TIFF) containing printed or handwritten text.
Enter comma-separated keywords to extract targeted information.
""")

uploaded_files = st.file_uploader(
    "Choose files (PDF, DOCX, TXT, PNG, JPG, JPEG, BMP, TIFF)",
    type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "bmp", "tiff"],
    accept_multiple_files=True
)

keywords = st.text_input(
    "Keywords (comma separated, optional)",
    placeholder="e.g., derivative, matrix, theorem"
)

if st.button("Extract Information"):
    if not uploaded_files:
        st.warning("Please upload at least one file.")
    elif not keywords:
        st.warning("Please enter some keywords.")
    else:
        with st.spinner("Processing files..."):
            try:
                files = [("files", (file.name, file.getvalue())) for file in uploaded_files]
                data = {"keywords": keywords}
                response = requests.post(
                    f"{BACKEND_URL}/upload",
                    files=files,
                    data=data,
                    timeout=180  # Increased timeout for large files/OCR
                )
                if response.status_code == 200:
                    result = response.json()
                    st.success("Extraction complete!")
                    if "data" in result and "results" in result["data"]:
                        for item in result["data"]["results"]:
                            st.markdown(f"### Keyword: `{item.get('keyword', 'N/A')}`")
                            for field in [
                                'definition', 'key_details', 'programs_courses',
                                'requirements', 'additional_context', 'relevant_quotes', 'relevant_text'
                            ]:
                                value = item.get(field)
                                if value:
                                    header = field.replace('_', ' ').title()
                                    st.markdown(f"**{header}:**")
                                    st.write(value)
                                    st.write("")
                            st.markdown("---")
                    else:
                        st.write(result)
                else:
                    try:
                        err = response.json()
                        st.error(f"Error: {err.get('error', response.text)}")
                    except Exception:
                        st.error(f"Backend error: {response.text}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.code(traceback.format_exc())

st.markdown("---")
st.caption("Supports PDF, DOCX, TXT, and image files (printed or handwritten). Handwriting OCR quality may vary based on Tesseract’s capabilities.")
