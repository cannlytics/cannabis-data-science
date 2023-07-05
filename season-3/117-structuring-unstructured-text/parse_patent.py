import openai
import pdfplumber
from pytesseract import image_to_string


# Check the number of pages.
report = pdfplumber.open('patent.pdf')
page_count = len(report.pages)

# Get the full text of the patent.
full_text = ''
for i, page in enumerate(report.pages[-5:]):


    # Save the first page as an image.
    image_file = './pages/page-%i.png' % i
    im = report.pages[0].to_image(resolution=90)
    im.save(image_file, format='PNG')

    # Read the text of the first page.
    page_text = image_to_string(image_file)
    full_text += page_text


# Close the report.
report.close()


# Create messages.
messages = [
    {'role': 'user', 'content': 'What are the morphological characteristics of this plant?' + full_text},
]

# Submit all prompts at the same time.
response = openai.ChatCompletion.create(
    model='gpt-4',
    temperature=0.0,
    max_tokens=4_000,
    messages=messages
)
content = response['choices'][0]['message']['content']  
print(content)
