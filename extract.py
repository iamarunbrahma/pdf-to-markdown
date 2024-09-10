import fitz  # PyMuPDF
import pdfplumber
import re
import json
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import os

# Load image captioning model
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load math to LaTeX conversion dictionary
with open("math_to_latex.json", "r") as f:
    math_to_latex = json.load(f)

# Define bullet point symbols
BULLET_POINTS = ['•', '◦', '▪', '▫', '●', '○']

def extract_tables(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages):
            page_tables = page.extract_tables()
            for table in page_tables:
                tables.append({"page": page_number, "content": table})
    return tables

def table_to_markdown(table):
    markdown = ""
    for i, row in enumerate(table):
        markdown += "| " + " | ".join(cell if cell else "" for cell in row) + " |\n"
        if i == 0:
            markdown += "|" + "|".join(["---"] * len(row)) + "|\n"
    return markdown

def caption_image(image):
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs.pixel_values

    generated_ids = model.generate(pixel_values, max_length=50)
    generated_caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    generated_caption = generated_caption.strip()
    return generated_caption

def convert_math_to_latex(text):
    for symbol, latex in math_to_latex.items():
        text = text.replace(symbol, f"${latex}$")
    return text

def is_bullet_point(text):
    return any(text.strip().startswith(bullet) for bullet in BULLET_POINTS)

def extract_markdown(pdf_path):
    doc = fitz.open(pdf_path)
    markdown_content = ""
    tables = extract_tables(pdf_path)
    table_index = 0
    current_list_item = ""
    in_list = False

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] == 0:  # Text
                block_rect = block["bbox"]
                page_height = page.rect.height
                if block_rect[1] < 50 or block_rect[3] > page_height - 50:
                    continue

                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        text = span["text"]
                        font_size = span["size"]
                        flags = span["flags"]

                        # # Skip headers and footers
                        # if font_size < 8 or font_size > 20:
                        #     continue

                        # Determine header level based on font size
                        if 16 < font_size <= 20:
                            text = f"## {text}\n\n"
                        elif 14 < font_size <= 16:
                            text = f"### {text}\n\n"
                        elif 12 < font_size <= 14:
                            text = f"#### {text}\n\n"
                        else:
                            # Apply formatting
                            if flags & 2**0:  # Superscript
                                text = f"<sup>{text}</sup>"
                            if flags & 2**1:  # Italic
                                text = f"*{text}*"
                            if flags & 2**3:  # Monospace
                                text = f"`{text}`"
                            if flags & 2**4:  # Bold
                                text = f"**{text}**"

                        # Convert mathematical symbols to LaTeX
                        text = convert_math_to_latex(text)

                        # Check for bullet points or numbered lists
                        if is_bullet_point(text) or re.match(r'^\d+\.', text.strip()):
                            if current_list_item:
                                markdown_content += current_list_item + "\n"
                            current_list_item = text.strip() + " "
                            in_list = True
                        elif in_list:
                            current_list_item += text + " "
                        else:
                            line_text += text + " "

                    if not in_list:
                        markdown_content += line_text.strip() + "\n"
                    elif not is_bullet_point(line_text) and not re.match(r'^\d+\.', line_text.strip()):
                        markdown_content += current_list_item + "\n"
                        current_list_item = ""
                        in_list = False

                if current_list_item:
                    markdown_content += current_list_item + "\n"
                    current_list_item = ""
                    in_list = False

                markdown_content += "\n"

            elif block["type"] == 1:  # Image
                image_rect = block["bbox"]
                pix = page.get_pixmap(clip=image_rect)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Save image
                image_filename = f"outputs/image_{page.number}_{block['number']}.png"
                image.save(image_filename)

                # Caption image
                caption = caption_image(image)

                # Add image to markdown
                markdown_content += f"![{caption}]({image_filename})\n\n"

        # Insert tables at their approximate positions
        while table_index < len(tables) and tables[table_index]["page"] == page.number:
            markdown_content += table_to_markdown(tables[table_index]["content"]) + "\n\n"
            table_index += 1

    # Post-processing
    markdown_content = re.sub(r'\*\s*\*', '', markdown_content)  # Remove empty bold/italic
    markdown_content = re.sub(r'\n{3,}', '\n\n', markdown_content)  # Remove excessive newlines
    markdown_content = re.sub(r'(\d+)\s*\n', '', markdown_content)  # Remove page numbers

    return markdown_content

def main(pdf_path):
    markdown_content = extract_markdown(pdf_path)
    
    with open("outputs/output.md", "w", encoding="utf-8") as f:
        f.write(markdown_content)

if __name__ == "__main__":
    pdf_path = "inputs/meta.pdf"
    main(pdf_path)