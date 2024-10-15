import os
import platform
import tempfile
import pytesseract
import docx2txt
from pdf2image import convert_from_path
import ftfy

# Function to set Tesseract path dynamically for Windows users
def set_tesseract_path():
    if platform.system() == "Windows":
        # Check both common installation paths for Tesseract on Windows
        program_files = os.getenv('ProgramFiles')
        program_files_x86 = os.getenv('ProgramFiles(x86)')
        
        possible_paths = [
            os.path.join(program_files, "Tesseract-OCR", "tesseract.exe"),
            os.path.join(program_files_x86, "Tesseract-OCR", "tesseract.exe")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break
        else:
            raise FileNotFoundError("Tesseract executable not found in standard Program Files directories. Please ensure Tesseract is installed.")
    else:
        # On non-Windows platforms, assume Tesseract is correctly installed in the system PATH
        if not shutil.which("tesseract"):
            raise FileNotFoundError("Tesseract is not installed or not in the system's PATH. Please install Tesseract.")

# Call the function to set the Tesseract path dynamically
set_tesseract_path()

# Function to handle file uploads (PDF/DOCX), convert PDF to images, and extract text from DOCX
def handle_file_upload(uploaded_file):
    if uploaded_file is None:
        return None, None

    file_type = uploaded_file.type
    text_content = ""

    # Handle PDF Files
    if file_type == "application/pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # Convert PDF to images
        images = convert_from_path(temp_file_path, dpi=300)

        # Apply OCR to extract text from each page using pytesseract
        ocr_texts = [pytesseract.image_to_string(image) for image in images]
        ocr_texts = [ftfy.fix_text(text) for text in ocr_texts]  # Clean up formatting issues

        return "\n".join(ocr_texts), images

    # Handle DOCX Files
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        text_content = docx2txt.process(temp_file_path)
        text_content = ftfy.fix_text(text_content)  # Clean up formatting issues
        text_content = " ".join(text_content.split())  # Remove excessive whitespace

        return text_content, None

    else:
        return None, f"Unsupported file type: {file_type}"
