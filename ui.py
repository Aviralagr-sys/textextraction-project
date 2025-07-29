import streamlit as st
import requests
import traceback

# Backend API URL (Flask server running on port 5001)
BACKEND_URL = "http://127.0.0.1:5001"

st.title("📄 Document & Image Information Extractor")
st.markdown("""
Upload PDFs, DOCX, TXT, or images (PNG, JPG, JPEG, BMP, TIFF) containing printed or handwritten text.
Enter comma-separated keywords to extract targeted information, or paste a snippet to search for context and pages.
""")

uploaded_files = st.file_uploader(
    "Choose files (PDF, DOCX, TXT, PNG, JPG, JPEG, BMP, TIFF)",
    type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "bmp", "tiff"],
    accept_multiple_files=True,
)

keywords = st.text_input(
    "Keywords (comma separated, optional)",
    placeholder="e.g., derivative, matrix, theorem"
)

snippet = st.text_area(
    "Or paste a few lines/snippet to find context and highlight page/image",
    height=120,
    placeholder="Enter text snippet to search"
)

col1, col2 = st.columns(2)

with col1:
    if st.button("Extract Information"):
        if not uploaded_files:
            st.warning("Please upload at least one file.")
        elif not keywords:
            st.warning("Please enter some keywords.")
        else:
            with st.spinner("Processing files..."):
                try:
                    files = [
                        ("files", (file.name, file.getvalue(), file.type if hasattr(file, 'type') else "application/octet-stream"))
                        for file in uploaded_files
                    ]
                    data = {"keywords": keywords}
                    response = requests.post(
                        f"{BACKEND_URL}/upload",
                        files=files,
                        data=data,
                        timeout=180,
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

                        if "highlighted_pages" in result and result["highlighted_pages"]:
                            st.subheader("🔍 Highlighted PDF Pages")
                            for img_name in result["highlighted_pages"]:
                                img_url = f"{BACKEND_URL}/highlighted_pages/{img_name}"
                                st.image(img_url, caption=img_name, use_container_width=True)
                    else:
                        try:
                            err = response.json()
                            st.error(f"Error: {err.get('error', response.text)}")
                        except Exception:
                            st.error(f"Backend error: {response.text}")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.code(traceback.format_exc())

with col2:
    if st.button("Search Context"):
        if not uploaded_files:
            st.warning("Please upload at least one file.")
        elif not snippet.strip():
            st.warning("Please enter a text snippet to search.")
        else:
            with st.spinner("Searching for context..."):
                try:
                    files = [
                        ("files", (file.name, file.getvalue(), file.type if hasattr(file, 'type') else "application/octet-stream"))
                        for file in uploaded_files
                    ]
                    data = {"snippet": snippet}
                    response = requests.post(
                        f"{BACKEND_URL}/search_context",
                        files=files,
                        data=data,
                        timeout=180,
                    )
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("status") == "success":
                            results = result.get("results", [])
                            if not results:
                                st.info("No matches found for your snippet.")

                            for item in results:
                                st.markdown(f"### File: `{item.get('file_name', 'N/A')}`" + (f", Page: {item.get('page')}" if item.get('page') else ""))
                                st.markdown(f"> {item.get('context', 'No context available')}")
                                st.markdown("---")

                            if "highlighted_pages" in result and result["highlighted_pages"]:
                                st.subheader("🔍 Highlighted Pages/Images")
                                for img_name in result["highlighted_pages"]:
                                    img_url = f"{BACKEND_URL}/highlighted_pages/{img_name}"
                                    st.image(img_url, caption=img_name, use_container_width=True)
                        else:
                            st.error("Error: " + str(result.get("error", "Unknown error.")))
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
st.caption("Supports PDF, DOCX, TXT, and image files (printed or handwritten). Handwriting OCR quality may vary based on Tesseract capabilities.")
