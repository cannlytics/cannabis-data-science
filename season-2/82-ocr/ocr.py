"""
Use OCR to Recognize PDF Text
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 8/30/2022
Updated: 9/7/2022
License: <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Description:

    Use OCR to recognize the text of other unrecognizable text in a CoA
    PDF. The PDF can then be parsed as normal with CoADoc. The process is:

        1. Convert all PDF pages to images, with ImageMagick.
        2. Convert each image to PDF, with recognizable text, with Tesseract.
        3. Compile the PDFs with text to a single PDF, with PyPDF2.

Setup:

    1. Install Tesseract.
    For Windows: <https://github.com/UB-Mannheim/tesseract/wiki>
    For all other OS: <https://github.com/madmaze/pytesseract>

    2. Install ImageMagick
    Download: <https://imagemagick.org/script/download.php#windows>
    Cloud: <https://stackoverflow.com/questions/43036268/do-i-have-access-to-graphicsmagicks-or-imagemagicks-in-a-google-cloud-function>

"""
# Standard imports.
import os

# External imports.
from PyPDF2 import PdfMerger
import pytesseract
from wand.image import Image
from wand.color import Color

# Internal imports.
from cannlytics.data.coas import CoADoc
from cannlytics.data.coas.cannalysis import parse_cannalysis_coa

# Specify where your data lives.
DATA_DIR = '.'
file_name = f'{DATA_DIR}/mist.pdf'
outfile = f'{DATA_DIR}/test.pdf'
temp_path = f'{DATA_DIR}/imgs'


def convert_pdf_to_images(filename, output_path, resolution=150):
    """ Convert a PDF into images. All the pages will be named:

        {pdf_filename}-{page_number}.png

    The function removes the alpha channel from the image and
    replace it with a white background.

    Authors: Thibaut Mattio, Averner <https://stackoverflow.com/a/42525093/5021266>
    License: CC BY-SA 4.0 <https://creativecommons.org/licenses/by-sa/4.0/>
    """
    image_files = []
    all_pages = Image(filename=filename, resolution=resolution)
    for i, page in enumerate(all_pages.sequence):
        with Image(page) as img:
            img.format = 'png'
            img.background_color = Color('white')
            img.alpha_channel = 'remove'
            image_filename = os.path.splitext(os.path.basename(filename))[0]
            image_filename = '{}-{}.png'.format(image_filename, i)
            image_filename = os.path.join(output_path, image_filename)
            img.save(filename=image_filename)
            image_files.append(image_filename)
    return image_files


# Create a directory to store images and rendered PDFs.
if not os.path.exists(temp_path): os.makedirs(temp_path)

# Convert each PDF page to an image.
image_files = convert_pdf_to_images(file_name, temp_path, resolution=300)

# Convert each image to PDF with text.
pdf_files = []
for image_file in image_files:
    pdf = pytesseract.image_to_pdf_or_hocr(image_file, extension='pdf')
    pdf_file = image_file.replace('.png', '.pdf')
    pdf_files.append(pdf_file)
    with open(pdf_file, 'w+b') as f:
        f.write(pdf)
    os.remove(image_file)

# Compile the PDFs with text to a single PDF.
merger = PdfMerger()
for pdf in pdf_files:
    merger.append(pdf)
merger.write(outfile)
merger.close()

# Remove individual PDF files.
for pdf in pdf_files:
    os.remove(pdf)

# Parse the CoA as normal!
parser = CoADoc()
lab = parser.identify_lims(outfile)
assert lab == 'Cannalysis'

# FIXME: This algorithm is throwing an error.
# data = parse_cannalysis_coa(parser, outfile)
# assert data is not None


#-----------------------------------------------------------------------
# Use CoADoc's built-in OCR.
#-----------------------------------------------------------------------

import re
import pdfplumber

# Pass a PDF through OCR.
parser = CoADoc()
doc = '../../.datasets/tests/210000043-Kush-Clouds-0.5g.pdf'
temp_path = '../../.datasets/tests/tmp'
temp_file = '../../.datasets/tests/tmp/ocr_coa.pdf'
parser.pdf_ocr(doc, temp_file, temp_path, resolution=180)

# Test that the PDF needs OCR.
report = pdfplumber.open(doc)
text = report.pages[0].extract_text()
if re.match(r'^\(cid:\d+\)', text):
    print('OCR needed.')
else:
    print('Text recognized.')

# Parse a PDF that requires OCR.
data = parser.parse_pdf(doc, temp_path=temp_path)
assert data is not None
