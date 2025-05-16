import base64
import requests

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
    prompt = f'''You are a document reconstruction assistant.
    Your task is to reconstruct a document from OCRed plain text and corresponding image information.
    Rules:
    Reformat the content using Markdown.
    Keep paragraphs together without splitting them unnecessarily.
    Ignore purely decorative graphical elements unless they carry important meaning.
    If the layout is unclear (for example, if a table is broken), reconstruct it in a way that makes logical sense based on the context.
    Try to preserve all important content.
    Do not hallucinate or add information not present in the original.
    Here's the OCR plain text:
    {text}
    '''

    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "X-Title": "ImagePromptApp",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openai/gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }
        ]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)

    if response.status_code != 200:
        raise Exception(f"API Error {response.status_code}: {response.text}")

    return response.json()["choices"][0]["message"]["content"]
