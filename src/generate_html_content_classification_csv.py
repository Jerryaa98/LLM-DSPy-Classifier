import os
import pandas as pd
from bs4 import BeautifulSoup


def extract_text_from_html(html_code, max_length=2000):
    """
    Extracts main text content from an HTML string, truncating to max_length chars.
    """
    try:
        soup = BeautifulSoup(html_code, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text[:max_length]
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""


def generate_html_content_classification_csv(html_txt_path, csv_path):
    """
    Reads a .txt file containing raw HTML code, extracts content, and creates a CSV for LLM classification.
    Columns: context (extracted text), question (fixed), answer (blank for manual labeling).
    """
    with open(html_txt_path, "r", encoding="utf-8") as f:
        html_code = f.read()

    context = extract_text_from_html(html_code)
    # Extract label from filename (expects format like html_1_yes.txt)
    label = os.path.splitext(os.path.basename(html_txt_path))[0].split('_')[-1]
    new_row = {
        "context": context,
        "question": "Is the following HTML page a website of a funding opportunity?",
        "answer": label  # Pre-fill with label from filename
    }

    # Ensure data directory exists
    data_dir = os.path.dirname(csv_path)
    os.makedirs(data_dir, exist_ok=True)

    # Append to CSV if it exists, else create new
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    df.to_csv(csv_path, index=False)
    print(f"CSV file updated at: {csv_path}")

if __name__ == "__main__":
    for html_txt_file in os.listdir(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "html")):
        html_txt_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "html", html_txt_file)
        csv_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "html_content_classification.csv")
        generate_html_content_classification_csv(html_txt_file, csv_file)
