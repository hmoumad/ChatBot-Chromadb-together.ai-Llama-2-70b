# Requirements:

# Before we start, make sure you have the following libraries installed:
    # 1. pdf2image: To convert PDF files into images. "pip install pdf2image"
    # 2. pytesseract: A Python wrapper for Google’s Tesseract OCR engine. "pip install pytesseract"
    # 3. OpenCV: For image preprocessing tasks like deskewing and grayscale conversion. "pip install opencv-python"
    # 4. pandas: For storing extracted text data in a structured manner. "pip install pandas"

    # attention the libraries convert_from_path and tesseract becouse you need to install some packages
    # for convert_from_path you need to install poppler you can click the link to know to install it "https://www.youtube.com/watch?v=SioLV0f0sr0&t=11s&ab_channel=CodingDiksha"
    # about tesseract you need to install tesseract you can click the link to know how "https://www.youtube.com/watch?v=LMM6s9JL5GA&ab_channel=datascienceAnywhere"

from pdf2image import convert_from_path
import pytesseract
import cv2
import numpy as np
import os

# function for preprocessing images
def deskew(image):
    if not isinstance(image, np.ndarray):
        raise ValueError("Input image is not a valid numpy array")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated

# function for extracting text from images
def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text


def extract_text_from_pdf(PATH_FILE):
    # function to transform images into text
    pages = convert_from_path(PATH_FILE, poppler_path="C:/Program Files/poppler-24.02.0/Library/bin")

    # Créer un dossier pour stocker les images si ce n'est pas déjà fait
    output_directory = 'test/img'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Sauvegarder les images des pages du PDF dans un répertoire
    for i, page in enumerate(pages):
        page.save(os.path.join(output_directory, 'page'+str(i)+'.jpg'), 'JPEG')

    extract_text = []
    for page in pages:
        # Step 2: Preprocess the image (deskew)
        preprocess_image = deskew(np.array(page))

        # Step 3: Extract text using OCR
        text = extract_text_from_image(preprocess_image)

        # step 4: add text page into list
        extract_text.append(text)

    return extract_text

# # Utilisation de la fonction extract_text_from_pdf
# PATH_FILE = "C:/Users/hamza/Desktop/Researches/biographie.pdf"
# extracted_text = extract_text_from_pdf(PATH_FILE)
# print(extracted_text)
