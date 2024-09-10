import fitz  # PyMuPDF
import pdfplumber
import re
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import os
import math

# Load image captioning model
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

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
    return generated_caption

def extract_markdown(pdf_path):
    doc = fitz.open(pdf_path)
    markdown_content = ""
    tables = extract_tables(pdf_path)
    table_index = 0

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] == 0:  # Text
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"]
                        font_size = span["size"]
                        flags = span["flags"]

                        # Determine header level based on font size
                        if font_size > 20:
                            markdown_content += f"# {text}\n\n"
                        elif font_size > 18:
                            markdown_content += f"## {text}\n\n"
                        elif font_size > 16:
                            markdown_content += f"### {text}\n\n"
                        elif font_size > 14:
                            markdown_content += f"#### {text}\n\n"
                        elif font_size > 12:
                            markdown_content += f"##### {text}\n\n"
                        else:
                            # Apply formatting
                            if flags & 2**0:  # Superscript
                                text = f"<sup>{text}</sup>"
                            if flags & 2**1:  # Italic
                                text = f"*{text}*"
                            if flags & 2**2:  # Serifed
                                pass  # No specific markdown for serifed
                            if flags & 2**3:  # Monospace
                                text = f"`{text}`"
                            if flags & 2**4:  # Bold
                                text = f"**{text}**"

                            markdown_content += text + " "

                markdown_content += "\n\n"

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

    return markdown_content

def main(pdf_path):
    markdown_content = extract_markdown(pdf_path)
    
    with open("outputs/output.md", "w", encoding="utf-8") as f:
        f.write(markdown_content)

if __name__ == "__main__":
    pdf_path = "inputs/meta.pdf"
    main(pdf_path)