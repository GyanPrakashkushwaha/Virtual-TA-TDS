import os
import sys
from dotenv import load_dotenv


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY5")
GEMINI_API_KEY2 = os.getenv("GEMINI_API_KEY2")
APIS_LIST = os.getenv('API_LIST')
OPEN_API_KEY = os.getenv('OPEN_API_KEY')
IMG_GENERATION_PROMPT = """Perform Optical Character Recognition (OCR) on the provided image to extract all readable text accurately. Follow these steps:

            1. Analyze the input image or document containing text
            2. Extract ALL visible text while preserving formatting, line breaks, and spatial relationships where possible
            3. Identify and transcribe any error messages, menu options, button labels, dialog boxes, and interface elements
            4. Verify the accuracy of recognized characters, especially for technical terms, error codes, and numerical values
            5. Present the extracted text in a clear, structured format that maintains the original context

            ## Output Requirements:
            - Return extracted text as plain, continuous text with natural spacing
            - Preserve original line breaks and formatting structure
            - Handle various fonts, sizes, and orientations of text
            - Be robust against noise, distortion, and low image quality
            - Flag any unclear or illegible parts with [UNCLEAR] markers
            - Do not include additional commentary or metadata
            """