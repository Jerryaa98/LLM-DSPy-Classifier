import os
import pandas as pd
import requests
from bs4 import BeautifulSoup

def fetch_website_text(url, max_length=2000):
    """
    Fetches the main text content from a webpage, truncating to max_length chars.
    """
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove scripts and styles
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text[:max_length]
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

def generate_link_content_csv(txt_path, csv_path):
    """
    Reads a .txt file with links, fetches their content, and creates a CSV for LLM classification.
    Columns: context (webpage text), question (fixed), answer (blank for manual labeling).
    """
    with open(txt_path, "r") as f:
        links = [line.strip() for line in f if line.strip()]

    data = []
    for link in links:
        print(f"Fetching: {link}")
        context = fetch_website_text(link)
        data.append({
            "context": context,
            "question": "Is the following link a website of a funding opportunity?",
            "answer": ""  # To be filled manually
        })

    # Ensure data directory exists
    data_dir = os.path.dirname(csv_path)
    os.makedirs(data_dir, exist_ok=True)

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"CSV file created at: {csv_path}")

if __name__ == "__main__":
    # Example usage
    txt_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "links.txt")
    csv_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "link_content_classification.csv")
    generate_link_content_csv(txt_file, csv_file)
