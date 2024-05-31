# Requirements:
# Before we start, make sure you have the following libraries installed:
    #install PyPDF-2 using command : "pip install pypdf2"

from PyPDF2 import PdfReader

def Extract_text_pypdf2(PATH_FILE_NORMAL):
    # Open the PDF file
    with open(PATH_FILE_NORMAL, 'rb') as file:
        # Create a PdfReader object
        reader = PdfReader(file)

        # List to store the content of all pages
        content = []

        # Extract text from each page
        for page in reader.pages:
            content.append(page.extract_text())

    return content


#PyPDF-2 can't exratact text from File scanned or images inside our PDF file
PATH_FILE_SCANED = "C:/Users/hamza/Desktop/Researches/Attestation.pdf"

#if the PDF File is normal, it means that we can copy it in this case PyPDF-2 can extract text
PATH_FILE_NORMAL = "D:/HAMZA M2/Dictionnary.pdf"

# content = Extract_text_pypdf2(PATH_FILE_NORMAL)
# print(content)
