from PIL import Image
import pytesseract

import config


def text_from_image(file_path: str) -> str:
    img = Image.open(file_path)
    text = pytesseract.image_to_string(img, lang='rus+eng')
    if config.ImageRecognitionConfig.UseGPT:
        return gpt_format(text, file_path)
    return text


def gpt_format(text: str, image_path: str) -> str:
    # TODO: implement this properly

    return ""