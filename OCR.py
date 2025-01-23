

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




# ------------------------------------------------------------------------------------------------------------------------


## EXTRACT TEXT METHOD 1


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



















# ------------------------------------------------------------------------------------------------------------------------


# # EXTRACT TEXT METHOD 2

# import fitz
# import io
# from PIL import Image
# import pytesseract


# # Set the correct path to tesseract.exe
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# def ocr_pdf_with_pymupdf(pdf_path, output_text_file="output_text.txt", zoom_x=2, zoom_y=2):
#     doc = fitz.open(pdf_path)
#     text_content = []
    
#     for page_index in range(len(doc)):
#         page = doc[page_index]
        
#         # Zoom parameters for higher resolution
#         mat = fitz.Matrix(zoom_x, zoom_y)
#         pix = page.get_pixmap(matrix=mat)
        
#         # Convert to PIL Image
#         img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
#         page_text = pytesseract.image_to_string(img)
#         text_content.append(f"--- Page {page_index+1} ---\n{page_text}\n")
    
#     # Save text
#     with open(output_text_file, "w", encoding="utf-8") as f:
#         f.write("\n".join(text_content))
    
#     print(f"OCR text saved to {output_text_file}")

# if __name__ == "__main__":
#     pdf_file = r"C:\Users\LEGION\Desktop\PDF extracter OCR\PublicWaterMassMailing.pdf"
#     ocr_pdf_with_pymupdf(pdf_file)



# ------------------------------------------------------------------------------------------------------------------------


# # # EXTRACT TABLES FROM PDF

# # Enhancing table detection using Hough Line Transform

# import fitz
# import cv2
# import numpy as np
# import pytesseract
# import pandas as pd
# import os
# from PIL import Image

# def extract_tables_hough(pdf_path, output_folder="extracted_tables"):
#     """Extracts tables from a scanned PDF using Hough Line Transform and saves as CSV."""
    
#     os.makedirs(output_folder, exist_ok=True)
#     doc = fitz.open(pdf_path)
    
#     table_count = 0
#     extracted_tables = []

#     for page_index in range(len(doc)):
#         page = doc[page_index]
#         pix = page.get_pixmap()
        
#         # Convert Pixmap to OpenCV Image
#         img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#         img = np.array(img)
#         gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#         # **Step 1: Preprocessing - Apply Adaptive Thresholding**
#         binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                        cv2.THRESH_BINARY_INV, 11, 2)

#         # **Step 2: Use Hough Line Transform to Detect Grid Lines**
#         edges = cv2.Canny(binary, 50, 150, apertureSize=3)
#         lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)

#         # Draw detected lines on a mask
#         mask = np.zeros_like(gray)
#         if lines is not None:
#             for line in lines:
#                 x1, y1, x2, y2 = line[0]
#                 cv2.line(mask, (x1, y1), (x2, y2), 255, 2)

#         # **Step 3: Combine Mask with Original Binary Image**
#         detected_tables = cv2.bitwise_and(binary, binary, mask=mask)

#         # **Step 4: Find Table Contours**
#         contours, _ = cv2.findContours(detected_tables, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         for idx, contour in enumerate(contours):
#             x, y, w, h = cv2.boundingRect(contour)

#             # **Step 5: Filter Out Small Noise & Keep Larger Tables**
#             if w > 100 and h > 50:  # Minimum size for valid tables
#                 detected_table = img[y:y+h, x:x+w]
#                 detected_table_path = os.path.join(output_folder, f"page{page_index+1}_table{idx+1}.png")
                
#                 # Save detected table images
#                 cv2.imwrite(detected_table_path, detected_table)

#                 # Extract Text from Detected Tables Using OCR
#                 table_text = pytesseract.image_to_string(detected_table, config="--psm 6")

#                 # Convert OCR Text to CSV Format
#                 table_data = [line.split() for line in table_text.split("\n") if line.strip()]
#                 if table_data:
#                     df = pd.DataFrame(table_data)
#                     table_csv_path = os.path.join(output_folder, f"page{page_index+1}_table{idx+1}.csv")
#                     df.to_csv(table_csv_path, index=False)
#                     extracted_tables.append(table_csv_path)
#                     table_count += 1

#     print(f"Total extracted tables saved as CSV: {table_count}")
#     return {
#         "extracted_table_count": table_count,
#         "saved_table_files": extracted_tables
#     }

# if __name__ == "__main__":
#     pdf_file = r"C:\Users\LEGION\Desktop\PDF extracter OCR\PublicWaterMassMailing.pdf"
#     extract_tables_hough(pdf_file)



# ------------------------------------------------------------------------------------------------------------------------





# import camelot

# def extract_tables_from_pdf(pdf_path, pages="1-end"):
#     tables = camelot.read_pdf(pdf_path, pages=pages)
    
#     print(f"Total tables found: {tables.n}")
    
#     for i, table in enumerate(tables):
#         # Export each table to CSV
#         csv_name = f"table_{i+1}.csv"
#         table.to_csv(csv_name)
#         print(f"Saved table {i+1} to {csv_name}")

# if __name__ == "__main__":
#     pdf_file = r"C:\Users\LEGION\Desktop\PDF extracter OCR\SIW Issue 416 28_04_2000 1.pdf"
#     extract_tables_from_pdf(pdf_file)


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
