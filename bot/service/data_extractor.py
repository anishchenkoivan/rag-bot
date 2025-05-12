from pathlib import Path
from bot.image.image_reader import text_from_image


def extract_data(data, source: str) -> str:
    if source == "str":
        return data

    file_path = data
    file_ext = Path(file_path).suffix.lower()
    if file_ext in (".txt", "docx", ""):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    if file_ext in ("jpg", "jpeg", "png"):
        return text_from_image(file_path)
    return ""
    # TODO: Add proper text/image recognition from PDF


def extract_questions(data, source: str) -> [str]:
    data = extract_data(data, source)
    return data.split("\n")


def format_question(questions: str) -> [str]:
    return questions.split("\n")
