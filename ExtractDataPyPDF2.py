# Requirements:
# Before we start, make sure you have the following libraries installed:
    #install PyPDF-2 using command : "pip install pypdf2"

from PyPDF2 import PdfReader

#PyPDF-2 can't exratact text from File scanned or images inside our PDF file
PATH_FILE_SCANED = "C:/Users/hamza/Desktop/Researches/Attestation.pdf"

#if the PDF File is normal, it means that we can copy it in this case PyPDF-2 can extract text
PATH_FILE_NORMAL = "C:/Users/hamza/Desktop/Researches/CV.pdf"
reader = PdfReader(PATH_FILE_NORMAL)

#affiche content of file
print(reader.pages)

# number of pages in file PDF
number_pages = len(reader.pages)
print(number_pages)

page_0 = reader.pages[0]
print(page_0)

#affiche text from PDF 
text = page_0.extract_text()
print(text)