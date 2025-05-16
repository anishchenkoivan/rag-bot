from pathlib import Path
from image.image_reader import text_from_image


def extract_data(data, source: str) -> str:
    if source == "str":
        return data

    file_path = data
    file_ext = Path(file_path).suffix.lower()
    if file_ext in (".txt", ".docx", ""):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    if file_ext in (".jpg", ".jpeg", ".png"):
        return text_from_image(file_path)
    return ""
    # TODO: Add proper text/image recognition from PDF


def extract_questions(data) -> list[str]:
    return data.split("\n")


def format_questions(questions: str) -> str:
    return "\n" + "\n".join([f"{i + 1}. {questions[i]}" for i in range(len(questions))])


def format_answers(answers: list[str]) -> str:
    return "\n" + "\n".join([f"{i + 1}. {answers[i]}" for i in range(len(answers))])
