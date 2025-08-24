import os
import pandas as pd

def generate_md_content_classification_csv(md_file_path, csv_path):
	"""
	Reads a .md file, extracts content, and creates/appends to a CSV for LLM classification.
	Columns: context (markdown), question (fixed), answer (from filename, as in the HTML script).
	"""
	with open(md_file_path, "r", encoding="utf-8") as f:
		md_content = f.read()

	# Extract label from filename (expects format like md_1_yes.md)
	label = os.path.splitext(os.path.basename(md_file_path))[0].split('_')[-1]
	new_row = {
		"context": md_content,
		"question": "Is the following Markdown page a website of a funding opportunity?",
		"answer": label
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
	md_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "md")
	csv_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "md_content_classification.csv")
	md_files = [f for f in os.listdir(md_dir) if f.endswith('.md')]
	md_files.sort()  # Sort filenames for consistent order
	# Remove CSV if it exists to avoid duplicate appends
	if os.path.exists(csv_file):
		os.remove(csv_file)
	for md_file in md_files:
		md_file_path = os.path.join(md_dir, md_file)
		generate_md_content_classification_csv(md_file_path, csv_file)
