
## EXTRACT IMAGES USING pymupdf (fitz)
import fitz
import cv2
import numpy as np
import os
from PIL import Image

def extract_images_from_scanned_pdf(pdf_path, output_folder="detected_images"):
    """Extracts only images, logos, or diagrams (NOT text) from a scanned PDF."""
    
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    
    image_count = 0

    for page_index in range(len(doc)):
        page = doc[page_index]
        pix = page.get_pixmap()
        
        # Convert Pixmap to OpenCV Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = np.array(img)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # **Step 1: Apply Edge Detection**
        edges = cv2.Canny(gray, 50, 200)

        # **Step 2: Find Contours of Possible Images**
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for idx, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)

            # **Step 3: Filter Out Small Text Regions & Keep Images**
            aspect_ratio = w / float(h)  # Aspect ratio to remove elongated text lines
            contour_area = w * h

            if w > 30 and h > 30 and 0.3 < aspect_ratio < 5.0 and contour_area > 1500:  # Image-like regions
                detected_img = img[y:y+h, x:x+w]
                detected_img_path = os.path.join(output_folder, f"page{page_index+1}_obj{idx+1}.png")

                # Save detected images
                cv2.imwrite(detected_img_path, detected_img)
                image_count += 1

    print(f"Total detected images/logos/diagrams extracted: {image_count}")

if __name__ == "__main__":
    pdf_file = r"C:\Users\LEGION\Desktop\PDF extracter OCR\SIW Issue 416 28_04_2000 1.pdf"
    extract_images_from_scanned_pdf(pdf_file)



## EXTRACT TEXT METHOD 1 (pytesseract)


import os
from pdf2image import convert_from_path
import pytesseract

def ocr_pdf_to_text(pdf_path, output_text_file="output_text.txt", dpi=300):
    # Convert PDF pages to images
    pages = convert_from_path(pdf_path, dpi=dpi)
    
    text_content = []
    
    for i, page in enumerate(pages):
        # OCR each page using pytesseract
        page_text = pytesseract.image_to_string(page)
        text_content.append(f"--- Page {i+1} ---\n{page_text}\n")
    
    # Write all OCR text to a file
    with open(output_text_file, "w", encoding="utf-8") as f:
        f.write("\n".join(text_content))

    print(f"Text OCR extraction completed. Saved to {output_text_file}")

if __name__ == "__main__":
    pdf_file = r"C:\Users\LEGION\Desktop\PDF extracter OCR\SIW Issue 416 28_04_2000 1.pdf"
    ocr_pdf_to_text(pdf_file)



## EXTRACT TEXT METHOD 2 : EASYOCR

## EASY OCR TEXT ONLY

import os
import numpy as np
from pdf2image import convert_from_path
import easyocr
import torch

def ocr_pdf_with_easyocr(pdf_path, output_text_file="output_text_easyOCR.txt", dpi=300, lang='en', use_gpu=True):
    """Extracts text from a scanned PDF using EasyOCR with optional GPU support."""

    # Check if GPU is available
    device = use_gpu and torch.cuda.is_available()  # Use GPU if available
    print(f"Using GPU: {device}")  # Show whether GPU is being used
    
    # Convert PDF to images
    pages = convert_from_path(pdf_path, dpi=dpi)
    
    # Initialize EasyOCR reader with GPU/CPU control
    reader = easyocr.Reader(['en'], gpu=device)
    
    text_content = []
    
    for i, page in enumerate(pages):
        print(f"Processing Page {i+1}...")
        
        # Convert PIL Image to NumPy array
        page_np = np.array(page)
        
        # Extract text using EasyOCR
        page_text = reader.readtext(page_np, detail=0)  # Extract text as a list
        
        # Store extracted text with page number
        text_content.append(f"--- Page {i+1} ---\n" + "\n".join(page_text) + "\n")
    
    # Save extracted text to file
    with open(output_text_file, "w", encoding="utf-8") as f:
        f.write("\n".join(text_content))

    print(f"OCR extraction completed. Text saved to {output_text_file}")

if __name__ == "__main__":
    pdf_file = r"C:\Users\LEGION\Desktop\PDF extracter OCR\SIW Issue 416 28_04_2000 1.pdf"  # Update PDF path
    ocr_pdf_with_easyocr(pdf_file)








## EXTRACT TABLES method 1 :

import camelot

def extract_tables_from_pdf(pdf_path, pages="1-end", flavor="lattice"):
    print(f"Processing PDF: {pdf_path}")
    print(f"Extracting tables using {flavor} method...")

    # Read tables using the specified flavor
    tables = camelot.read_pdf(pdf_path, pages=pages, flavor=flavor)

    print(f"Total tables found: {tables.n}")

    if tables.n == 0 and flavor == "lattice":
        print("No tables detected with 'lattice' mode. Trying 'stream' mode...")
        tables = camelot.read_pdf(pdf_path, pages=pages, flavor="stream")

    for i, table in enumerate(tables):
        csv_name = f"table_{i+1}_{flavor}.csv"
        table.to_csv(csv_name)
        print(f"Saved table {i+1} to {csv_name}")

if __name__ == "__main__":
    pdf_file = r"C:\Users\LEGION\Desktop\PDF extracter OCR\SIW Issue 416 28_04_2000 1.pdf"

    # Run both lattice and stream modes for better accuracy
    extract_tables_from_pdf(pdf_file, pages="1-end", flavor="lattice")
    extract_tables_from_pdf(pdf_file, pages="1-end", flavor="stream")



# EXTRACT TABLES method 2 :

import tabula

pdf_path = r"C:\Users\LEGION\Desktop\PDF extracter OCR\SIW Issue 416 28_04_2000 1.pdf"

dfs = tabula.read_pdf(pdf_path,pages="all", encoding="ISO-8859-1")

for i in range(len(dfs)):
    dfs[i].to_csv(f"table{i}.csv")
