import sys
import os
import html2text

def html_txt_to_markdown_dir(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            input_txt_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            output_md_path = os.path.join(output_dir, base_name + '.md')
            with open(input_txt_path, 'r', encoding='utf-8') as f:
                html_code = f.read()
            markdown = html2text.html2text(html_code)
            with open(output_md_path, 'w', encoding='utf-8') as f:
                f.write(markdown)
            print(f"Converted {input_txt_path} to {output_md_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python html2md.py input_dir output_dir")
        sys.exit(1)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    html_txt_to_markdown_dir(input_dir, output_dir)