import os
import pandas as pd

def generate_link_classification_csv(txt_path, csv_path):
    """
    Reads a .txt file with links and creates a CSV for LLM-based classification.
    Columns: context (the link), question (fixed), answer (blank for manual labeling).
    """
    # Read links from txt file
    with open(txt_path, "r") as f:
        links = [line.strip() for line in f if line.strip()]

    # Prepare data for CSV
    data = []
    for link in links:
        data.append({
            "context": link,
            "question": f"Is the following link: {link} a website of a funding opportunity?",
            "answer": ""  # To be filled manually
        })

    # Ensure data directory exists
    data_dir = os.path.dirname(csv_path)
    os.makedirs(data_dir, exist_ok=True)

    # Write to CSV
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"CSV file created at: {csv_path}")

if __name__ == "__main__":
    # Example usage
    txt_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "links.txt")
    csv_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "link_classification.csv")
    generate_link_classification_csv(txt_file, csv_file)