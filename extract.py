import fitz  # PyMuPDF
import pdfplumber
import re
import json
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import os
import logging
from pathlib import Path

log_file = f"{Path(__file__).stem}.log"
logging.basicConfig(
    level=logging.INFO,  # Set log level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),  # Log to file
        logging.StreamHandler()         # Log to console
    ]
)
logger = logging.getLogger(__name__)

# Load image captioning model
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define bullet point symbols
BULLET_POINTS = '•◦▪▫●○'

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

    generated_ids = model.generate(pixel_values, max_length=30)
    generated_caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    generated_caption = generated_caption.strip()
    return generated_caption


def clean_text(text):
    # Remove leading/trailing whitespaces
    text = text.strip()
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text

def apply_formatting(text, flags):
    # Apply formatting without leading/trailing spaces
    text = text.strip()
    if not text:
        return text
    
    is_bold = flags & 2**4
    is_italic = flags & 2**1
    is_monospace = flags & 2**3
    is_superscript = flags & 2**0
    is_subscript = flags & 2**5 

    if is_monospace:
        text = f"`{text}`"
    elif is_superscript and not bool(re.search(r'\s+', text)):
        text = f"^{text}^"
    elif is_subscript and not bool(re.search(r'\s+', text)):
        text = f"~{text}~"

    if is_bold and is_italic:
        text = f"***{text}***"
    elif is_bold:
        text = f"**{text}**"
    elif is_italic:
        text = f"*{text}*"

    text = f" {text} "
    
    return text

def is_bullet_point(text):
    return text.strip().startswith(tuple(BULLET_POINTS))

def convert_bullet_to_markdown(text):
    # Remove any leading newlines and spaces, then add the markdown bullet
    text = re.sub(r'^\s*', '', text)
    return re.sub(f'^[{re.escape(BULLET_POINTS)}]\s*', '- ', text)

def is_numbered_list_item(text):
    # Check if the text starts with a number followed by a dot or parenthesis
    return bool(re.match(r'^\d+\s{0,3}[.)]', text.strip()))

def convert_numbered_list_to_markdown(text, list_counter):
    text = re.sub(r'^\s*', '', text)
    return re.sub(r'^\d+\s{0,3}[.)]', f"{list_counter}. ", text)

def is_horizontal_line(text):
    # Check if the text consists only of underscores or dashes
    return bool(re.match(r'^[_-]+$', text.strip()))

def extract_links(page):
    links = []
    for link in page.get_links():
        if link["kind"] == 2:  # URI link
            links.append({
                "rect": link["from"],
                "uri": link["uri"]
            })
    return links

def detect_code_block(text):
    patterns = {
        'python': r'(?s)^(?:(?:from|import|def|class|if|for|while|try|except|with)\s|\s{4})',
        'javascript': r'(?s)^(?:function|const|let|var|if|for|while|try|catch|class)\s',
        'html': r'(?s)^<(!DOCTYPE|html|head|body|div|p|a|script|style)',
        'shell': r'(?s)^(?:\$|\#)\s',
        'bash': r'(?s)^(?:#!/bin/bash|alias|export|source|echo|read|if|for|while|case|function)',
    }
    
    for lang, pattern in patterns.items():
        if re.match(pattern, text, re.MULTILINE):
            return lang
    return None

def extract_markdown(pdf_path):
    doc = fitz.open(pdf_path)
    markdown_content = ""
    tables = extract_tables(pdf_path)
    table_index = 0
    list_counter = 0  # Counter for numbered lists
    in_code_block = False
    code_block_content = ""
    code_block_lang = None

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        page_height = page.rect.height
        links = extract_links(page)

        for block in blocks:
            if block["type"] == 0:  # Text
                # Skip headers and footers
                block_rect = block["bbox"]
                if block_rect[1] < 50 or block_rect[3] > page_height - 50:
                    continue

                block_text = ""
                last_y1 = None
                last_font_size = None

                for line in block["lines"]:
                    line_text = ""
                    curr_font_size = [span["size"] for span in line["spans"]]

                    for span in line["spans"]:
                        text = span["text"]
                        font_size = span["size"]
                        flags = span["flags"]
                        span_rect = span["bbox"]

                        # Check for horizontal line
                        if is_horizontal_line(text):
                            line_text += "\n---\n"  # Add horizontal line in Markdown
                            continue

                        # Determine header level based on font size
                        header_level = 1 if font_size > 24 else 2 if font_size > 20 else 3 if font_size > 18 else 4 if font_size > 16 else 5 if font_size > 14 else 6 if font_size > 12 else 0

                        if header_level > 0:
                            text = f"\n{'#' * header_level} {clean_text(text)}\n\n"
                        else:
                            # Apply formatting
                            is_list_item = is_bullet_point(text) or is_numbered_list_item(text)

                            if is_list_item:
                                # If it's a list item, split the marker and content
                                marker, content = re.split(r'(?<=^[•◦▪▫●○\d.)])\s*', text, 1)
                                # Apply formatting only to content
                                formatted_content = apply_formatting(content, flags)
                                text = f"{marker} {formatted_content}"
                            else:
                                # If not a list item, apply formatting to entire text
                                text = apply_formatting(text, flags)
                        
                        # Check if the span intersects with any link
                        for link in links:
                            if fitz.Rect(span_rect).intersects(link["rect"]):
                                text = f"[{text.strip()}]({link['uri']})"
                                break

                        line_text += text

                    # Check if this is a new line or continuation of the previous line
                    if last_y1 is not None:
                        avg_last_font_size = sum(last_font_size) / len(last_font_size) if last_font_size else 0
                        avg_current_font_size = sum(curr_font_size) / len(curr_font_size)
                        font_size_changed = abs(avg_current_font_size - avg_last_font_size) > 1

                        if abs(line["bbox"][3] - last_y1) > 2 or font_size_changed:
                            block_text += "\n"
                    
                    block_text += clean_text(line_text) + " "
                    last_font_size = curr_font_size
                    last_y1 = line["bbox"][3]  # Bottom y-coordinate of the current line

                # Process block text for bullet points and numbered lists
                lines = block_text.split('\n')
                for i, line in enumerate(lines):
                    clean_line = clean_text(line)

                    if not in_code_block:
                        code_lang = detect_code_block(clean_line)
                        if code_lang:
                            in_code_block = True
                            code_block_lang = code_lang
                            code_block_content = clean_line + "\n"
                            continue

                    if in_code_block:
                        code_block_content += clean_line + "\n"
                        if i == len(lines) - 1 or detect_code_block(lines[i+1]) != code_block_lang:
                            markdown_content += f"```{code_block_lang}\n{code_block_content}```\n\n"
                            in_code_block = False
                            code_block_content = ""
                            code_block_lang = None

                    else:
                        if is_bullet_point(clean_line):
                            markdown_content += "\n" + convert_bullet_to_markdown(clean_line)
                            list_counter = 0  # Reset numbered list counter
                        elif is_numbered_list_item(clean_line):
                            list_counter += 1
                            markdown_content += "\n" + convert_numbered_list_to_markdown(clean_line, list_counter)   
                        else:
                            markdown_content += f"{clean_line}\n"
                            list_counter = 0  # Reset numbered list counter

                markdown_content += "\n"

            elif block["type"] == 1:  # Image
                image_rect = block["bbox"]
                pix = page.get_pixmap(clip=image_rect)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Save image
                image_filename = f"outputs/image_{int(page.number)+1}_{block['number']}.png"
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
    markdown_content = re.sub(r'\n{3,}', '\n\n', markdown_content)  # Remove excessive newlines
    markdown_content = re.sub(r'(\d+)\s*\n', '', markdown_content)  # Remove page numbers
    markdown_content = re.sub(r' +', ' ', markdown_content)  # Remove multiple spaces
    markdown_content = re.sub(r'\s*(---\n)+', '\n\n---\n', markdown_content)  # Remove duplicate horizontal lines

    return markdown_content

def main(pdf_path):
    # Ensure output directory exists
    os.makedirs("outputs", exist_ok=True)

    markdown_content = extract_markdown(pdf_path)
    
    with open("outputs/output.md", "w", encoding="utf-8") as f:
        f.write(markdown_content)

    print(f"Markdown content has been saved to outputs/output.md")

if __name__ == "__main__":
    pdf_path = "inputs/attention.pdf"
    main(pdf_path)