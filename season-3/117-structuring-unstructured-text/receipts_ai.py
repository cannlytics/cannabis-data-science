"""
BudSpender | Cannabis Receipt Parser
Copyright (c) 2023 Cannlytics

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
    Candace O'Sullivan-Sutherland <https://github.com/candy-o>
Created: 4/24/2023
Updated: 6/16/2023
License: <https://github.com/cannlytics/cannlytics/blob/main/LICENSE>
"""
# Standard imports:
from datetime import datetime
import json
import gc
import re
from typing import List, Optional

# External imports:
from cannlytics import __version__
from cannlytics.ai import (
    AI_WARNING,
    INSTRUCTIONAL_PROMPT,
    MAX_PROMPT_LENGTH,
    gpt_to_json,
)
from cannlytics.data import create_hash
import cv2
import pandas as pd
import pdfplumber
import pytesseract


# Prompt to parse data from a cannabis receipt.
RECEIPTS_PROMPT = """Given text, extract JSON, where:

| Field | Example | Description |
|-------|---------|-------------|
| `date_sold` | "2020-04-20" | The date the receipt was sold. |
| `invoice_number` | "123456789" | The receipt number. |
| `product_names` | ["Blue Rhino Pre-Roll"] | The names of the product purchased. |
| `product_types` | ["flower"] | The types of the products purchased. |
| `product_quantities` | [1] | The quantities of the products purchased. |
| `product_prices` | [5.0] | The prices of the products purchased. |
| `product_ids` | ["5f8b9c4b0f5c4b0008d1b2b0"] | The IDs of the products purchased. |
| `total_amount` | 5.0 | The total amount of all product prices. |
| `subtotal` | 5.0 | The subtotal of the receipt. |
| `total_discount` | 0.0 | The amount of discount applied to the transaction, if applicable. |
| `total_paid` | 5.0 | The total amount paid. |
| `change_due` | 0.0 | The amount of change due. |
| `rewards_earned` | 0.0 | The amount of rewards earned. |
| `rewards_spent` | 0.0 | The amount of rewards spent. |
| `total_rewards` | 0.0 | The total amount of rewards. |
| `city_tax` | 0.0 | The amount of city tax applied to the transaction, if applicable. |
| `county_tax` | 0.0 | The amount of county tax applied to the transaction, if applicable. |
| `state_tax` | 0.0 | The amount of state tax applied to the transaction, if applicable. |
| `excise_tax` | 0.0 | The amount of excise tax applied to the transaction, if applicable. |
| `retailer` | "BudHouse" | The name of the retailer. |
| `retailer_license_number` | "C11-0000001-LIC" | The license number of the retailer. |
| `retailer_address` | "1234 Main St, San Diego, CA 92101" | The address of the retailer. |
| `budtender` | "John Doe" | The name of the budtender. |
"""


class ReceiptsParser(object):
    """A cannabis receipt parser, powered by OpenAI."""

    def __init__(
            self,
            default_config: Optional[str] = '--oem 3 --psm 6',
            openai_api_key: Optional[str] = None,
            model: Optional[str] = 'gpt-4',
            base_prompt: Optional[str] = RECEIPTS_PROMPT,
            instructions: Optional[str] = INSTRUCTIONAL_PROMPT,
            max_tokens: Optional[int] = MAX_PROMPT_LENGTH,
        ) -> None:
        """Initialize an Open Data API client.
        Args:
            default_config: The default configuration for OpenCV.
            openai_api_key: The OpenAI API key.
            model: The OpenAI model to use, 'gpt-4' by default.
            base_prompt: The base prompt to use for the OpenAI model.
        """
        # Parameters.
        self.default_config = default_config
        self.openai_api_key = openai_api_key
        self.model = model
        self.base_prompt = base_prompt
        self.instructions = instructions
        self.max_tokens = max_tokens

        # State.
        self.total_cost = 0
        self.img = None
        self.image_text = ''
        self.extracted_data = None
    
    def image_to_pdf_to_text(self, image_file: str) -> str:
        """Extract the text from an image by converting it to a PDF.
        Args:
            image_file: The path to the image file.
        """
        img = self.rotate(cv2.imread(str(image_file), cv2.IMREAD_COLOR))
        pdf = pytesseract.image_to_pdf_or_hocr(img, extension='pdf')
        pdf_file = image_file.replace('.jpg', '.pdf')
        with open(pdf_file, 'w+b') as f:
            f.write(pdf)
            f.close()
        report = pdfplumber.open(pdf_file)
        text = ''
        for page in report.pages:
            text += page.extract_text()
        return text
    
    def image_to_text(
            self,
            image_file: str,
            median_blur: Optional[int] = 0
        ) -> str:
        """Extract the text from an image.
        Args:
            image_file: The path to the image file.
            median_blur: Removes noise. Must be a positive odd integer.
        """
        img = self.rotate(cv2.imread(str(image_file), cv2.IMREAD_COLOR))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if median_blur: img = cv2.medianBlur(img, median_blur)
        text = pytesseract.image_to_string(img, config=self.default_config)
        return text

    def rotate(self, image, center=None, scale=1.0):
        """Rotate an image with text into a readable position, as well as possible.
        Credit: Mousam Singh <https://stackoverflow.com/a/55122911/5021266>
        License: CC BY-SA 4.0 <https://creativecommons.org/licenses/by-sa/4.0/>
        """
        (h, w) = image.shape[:2]
        if center is None:
            center = (w / 2, h / 2)
        degrees = int(re.search(
            '(?<=Rotate: )\d+',
            pytesseract.image_to_osd(image),
        ).group(0))
        angle = 360 - degrees
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated
    
    def parse(
            self,
            doc: str,
            openai_api_key: Optional[str] = None,
            model: Optional[str] = 'gpt-4',
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = 0.0,
            system_prompts: Optional[List[str]] = None,
            verbose: Optional[bool] = False,
            user: Optional[str] = 'cannlytics',
            retry_pause: Optional[float] = 3.33,
        ) -> dict:
        """Parse a receipt with OpenAI's GPT model and return the data as JSON."""
        # Initialize parameters.
        if max_tokens is None:
            max_tokens = self.max_tokens

        # Get the text of a receipt.
        text = self.image_to_text(doc)

        # Begin the observation with the hash of the text.
        obs = {}
        obs['hash'] = create_hash(text, private_key='')

        # Format the system prompts.
        if not system_prompts:
            system_prompts = [self.instructions, self.base_prompt]

        # Initialize OpenAI.
        if openai_api_key is None:
            openai_api_key = self.openai_api_key
        if model is None:
            model = self.model

        # Parse the receipt.
        extracted_data, cost = gpt_to_json(
            text,
            system_prompts=system_prompts,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            user=user,
            openai_api_key=openai_api_key,
            verbose=verbose,
            retry_pause=retry_pause,
        )
        obs = {**obs, **extracted_data}

        # Calculate total price.
        prices = obs.get('product_prices', [])
        obs['total_price'] = 0
        for price in prices:
            try:
                obs['total_price'] += float(price)
            except:
                pass

        # Calculate total tax.
        obs['total_tax'] = 0
        tax_keys = [key for key in obs.keys() if '_tax' in key]
        for key in tax_keys:
            try:
                obs['total_tax'] += float(obs[key])
            except:
                pass

        # Calculate total transactions.
        obs['total_transactions'] = len(prices)

        # Future work: Augment retailer data.

        # Future work: Augment product data with product ID.

        # Future work: Save a copy of the image to Firebase Storage.

        # Mint the observation with unique IDs.
        obs['algorithm'] = 'receipts_ai.py'
        obs['algorithm_entry_point'] = 'parse_receipt_with_ai'
        obs['algorithm_version'] = __version__
        obs['parsed_at'] = datetime.now().isoformat()
        obs['warning'] = AI_WARNING
        self.total_cost += cost
        return obs
    
    def save(self, obs, filename):
        """Save a receipt to a file."""
        if filename.endswith('json'):
            with open(filename, 'w') as f:
                json.dump(obs, f, indent=4)
                f.close()
        elif filename.endswith('csv'):
            pd.DataFrame([obs]).to_csv(filename)
        elif filename.endswith('xlsx'):
            pd.DataFrame([obs]).to_excel(filename)
        else:
            raise ValueError(f'Unknown file type: {filename}')
        
    def quit(self):
        """Reset the parser and perform garbage cleaning."""
        self.model = 'gpt-4'
        self.openai_api_key = None
        self.total_cost = 0
        self.img = None
        self.image_text = ''
        self.extracted_data = None
        gc.collect()


# === Tests ===
if __name__ == '__main__':

    # Initialize a receipt parser.

    # Initialize OpenAI.
    from dotenv import dotenv_values
    config = dotenv_values('../../.env')
    openai_api_key = config['OPENAI_API_KEY']

    # Initialize a receipt parser.
    parser = ReceiptsParser(
        model='gpt-4',
        openai_api_key=openai_api_key,
    )

    # [✓] TEST: Parse a receipt.
    image_file = 'receipt-1.jpg'
    outfile = 'receipt-1.xlsx'
    receipt_data = parser.parse(image_file)
    print('Cost:', parser.total_cost)
    parser.save(receipt_data, outfile)

    # # [ ] TEST: Parse a folder of receipts.
    # RECEIPT_DIR = '.datasets/receipts/tests/'
    # all_receipt_data = []
    # file_types = ['.jpg', '.png', '.jpeg']
    # filenames = os.listdir(RECEIPT_DIR)
    # for filename in filenames:
    #     if any(filename.endswith(file_type) for file_type in file_types):
    #         image_file = os.path.join(RECEIPT_DIR, filename)
    #         receipt_data = parser.parse(image_file)
    #         all_receipt_data.append(receipt_data)
