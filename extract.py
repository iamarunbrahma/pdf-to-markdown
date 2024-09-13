import fitz  # PyMuPDF
import pdfplumber
import re
import pytesseract
import cv2
import numpy as np
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import os
import logging
import traceback
import warnings
from pathlib import Path
from abc import ABC, abstractmethod
import argparse

warnings.filterwarnings("ignore")

class PDFExtractor(ABC):
    """Abstract base class for PDF extraction."""

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.setup_logging()

    def setup_logging(self):
        """Set up logging configuration."""
        log_file = f"{Path(__file__).stem}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def extract(self):
        """Abstract method for extracting content from PDF."""
        pass

class MarkdownPDFExtractor(PDFExtractor):
    """Class for extracting markdown-formatted content from PDF."""

    BULLET_POINTS = '•◦▪▫●○'

    def __init__(self, pdf_path):
        super().__init__(pdf_path)
        self.pdf_filename = Path(pdf_path).stem
        self.setup_image_captioning()

    def setup_image_captioning(self):
        """Set up the image captioning model."""
        try:
            self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self.feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.logger.info("Image captioning model set up successfully.")
        except Exception as e:
            self.logger.error(f"Error setting up image captioning model: {e}")
            self.logger.exception(traceback.format_exc())

    def extract(self):
        """Extract markdown content from PDF."""
        try:
            markdown_content = self.extract_markdown()
            self.save_markdown(markdown_content)
            self.logger.info(f"Markdown content has been saved to outputs/{self.pdf_filename}.md")
        except Exception as e:
            self.logger.error(f"Error processing PDF: {e}")
            self.logger.exception(traceback.format_exc())

    def extract_markdown(self):
        """Main method to extract markdown from PDF."""
        try:
            doc = fitz.open(self.pdf_path)
            markdown_content = ""
            tables = self.extract_tables()
            table_index = 0
            list_counter = 0
            in_code_block = False
            code_block_content = ""
            code_block_lang = None
            prev_line = ""

            for page_num, page in enumerate(doc):
                self.logger.info(f"Processing page {page_num + 1}")
                blocks = page.get_text("dict")["blocks"]
                page_height = page.rect.height
                links = self.extract_links(page)

                for block in blocks:
                    if block["type"] == 0:  # Text
                        markdown_content += self.process_text_block(block, page_height, links, list_counter, in_code_block, code_block_content, code_block_lang, prev_line)
                    elif block["type"] == 1:  # Image
                        markdown_content += self.process_image_block(page, block)

                # Insert tables at their approximate positions
                while table_index < len(tables) and tables[table_index]["page"] == page.number:
                    markdown_content += "\n\n" + self.table_to_markdown(tables[table_index]["content"]) + "\n\n"
                    table_index += 1

            return self.post_process_markdown(markdown_content)
        except Exception as e:
            self.logger.error(f"Error extracting markdown: {e}")
            self.logger.exception(traceback.format_exc())
            return ""

    def extract_tables(self):
        """Extract tables from PDF using pdfplumber."""
        tables = []
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_number, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    for table in page_tables:
                        tables.append({"page": page_number, "content": table})
            self.logger.info(f"Extracted {len(tables)} tables from the PDF.")
        except Exception as e:
            self.logger.error(f"Error extracting tables: {e}")
            self.logger.exception(traceback.format_exc())
        return tables

    def table_to_markdown(self, table):
        """Convert a table to markdown format."""
        if not table:
            return ""

        try:
            table = [['' if cell is None else str(cell).strip() for cell in row] for row in table]
            col_widths = [max(len(cell) for cell in col) for col in zip(*table)]

            markdown = ""
            for i, row in enumerate(table):
                formatted_row = [cell.ljust(col_widths[j]) for j, cell in enumerate(row)]
                markdown += "| " + " | ".join(formatted_row) + " |\n"

                if i == 0:
                    markdown += "|" + "|".join(["-" * (width + 2) for width in col_widths]) + "|\n"

            return markdown
        except Exception as e:
            self.logger.error(f"Error converting table to markdown: {e}")
            self.logger.exception(traceback.format_exc())
            return ""

    def perform_ocr(self, image):
        """Perform OCR on the given image."""
        try:
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            ocr_result = pytesseract.image_to_data(opencv_image, output_type=pytesseract.Output.DICT)
            
            result = ""
            for word in ocr_result['text']:
                if word.strip() != "":
                    result += word + " "

                if len(result) > 30:
                    break
            
            return result.strip()
        except Exception as e:
            self.logger.error(f"Error performing OCR: {e}")
            self.logger.exception(traceback.format_exc())
            return ""

    def caption_image(self, image):
        """Generate a caption for the given image."""
        try:
            ocr_text = self.perform_ocr(image)
            if ocr_text:
                return ocr_text
            
            inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
            pixel_values = inputs.pixel_values

            generated_ids = self.model.generate(pixel_values, max_length=30)
            generated_caption = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return generated_caption.strip()
        except Exception as e:
            self.logger.error(f"Error captioning image: {e}")
            self.logger.exception(traceback.format_exc())
            return "Image"

    def clean_text(self, text):
        """Clean the given text by removing extra spaces."""
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    def apply_formatting(self, text, flags):
        """Apply markdown formatting to the given text based on flags."""
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

        return f" {text} "

    def is_bullet_point(self, text):
        """Check if the given text is a bullet point."""
        return text.strip().startswith(tuple(self.BULLET_POINTS))

    def convert_bullet_to_markdown(self, text):
        """Convert a bullet point to markdown format."""
        text = re.sub(r'^\s*', '', text)
        return re.sub(f'^[{re.escape(self.BULLET_POINTS)}]\s*', '- ', text)

    def is_numbered_list_item(self, text):
        """Check if the given text is a numbered list item."""
        return bool(re.match(r'^\d+\s{0,3}[.)]', text.strip()))

    def convert_numbered_list_to_markdown(self, text, list_counter):
        """Convert a numbered list item to markdown format."""
        text = re.sub(r'^\s*', '', text)
        return re.sub(r'^\d+\s{0,3}[.)]', f"{list_counter}. ", text)

    def is_horizontal_line(self, text):
        """Check if the given text represents a horizontal line."""
        return bool(re.match(r'^[_-]+$', text.strip()))

    def extract_links(self, page):
        """Extract links from the given page."""
        links = []
        try:
            for link in page.get_links():
                if link["kind"] == 2:  # URI link
                    links.append({
                        "rect": link["from"],
                        "uri": link["uri"]
                    })
            self.logger.info(f"Extracted {len(links)} links from the page.")
        except Exception as e:
            self.logger.error(f"Error extracting links: {e}")
            self.logger.exception(traceback.format_exc())
        return links

    def detect_code_block(self, prev_line, current_line):
        """Detect if the current line starts a code block."""
        patterns = {
            'python': [
                (r'^(?:from|import)\s+\w+', r'^(?:from|import|def|class|if|for|while|try|except|with)\s'),
                (r'^(?:def|class)\s+\w+', r'^\s{4}'),
                (r'^\s{4}', r'^\s{4,}')
            ],
            'javascript': [
                (r'^(?:function|const|let|var)\s+\w+', r'^(?:function|const|let|var|if|for|while|try|catch|class)\s'),
                (r'^(?:if|for|while)\s*\(', r'^\s{2,}'),
                (r'^\s{2,}', r'^\s{2,}')
            ],
            'html': [
                (r'^<(!DOCTYPE|html|head|body|div|p|a|script|style)', r'^<(!DOCTYPE|html|head|body|div|p|a|script|style)'),
                (r'^<\w+.*>$', r'^\s{2,}<'),
                (r'^\s{2,}<', r'^\s{2,}<')
            ],
            'shell': [
                (r'^(?:\$|\#)\s', r'^(?:\$|\#)\s'),
                (r'^[a-z_]+\s*=', r'^[a-z_]+\s*=')
            ],
            'bash': [
                (r'^(?:#!/bin/bash|alias|export|source)\s', r'^(?:#!/bin/bash|alias|export|source|echo|read|if|for|while|case|function)\s'),
                (r'^(?:if|for|while|case|function)\s', r'^\s{2,}'),
                (r'^\s{2,}', r'^\s{2,}')
            ],
            'cpp': [
                (r'^#include\s*<', r'^(?:#include|using|namespace|class|struct|enum|template|typedef)\s'),
                (r'^(?:class|struct|enum)\s+\w+', r'^\s{2,}'),
                (r'^\s{2,}', r'^\s{2,}')
            ],
            'java': [
                (r'^(?:import|package)\s+\w+', r'^(?:import|package|public|private|protected|class|interface|enum)\s'),
                (r'^(?:public|private|protected)\s+class\s+\w+', r'^\s{4,}'),
                (r'^\s{4,}', r'^\s{4,}')
            ],
            'json': [
                (r'^\s*{', r'^\s*["{[]'),
                (r'^\s*"', r'^\s*["}],?$'),
                (r'^\s*\[', r'^\s*[}\]],?$')
            ]
        }
        
        for lang, pattern_pairs in patterns.items():
            for prev_pattern, curr_pattern in pattern_pairs:
                if (re.match(prev_pattern, prev_line.strip()) and 
                    re.match(curr_pattern, current_line.strip())):
                    return lang
                
        return None

    def process_text_block(self, block, page_height, links, list_counter, in_code_block, code_block_content, code_block_lang, prev_line):
        """Process a text block and convert it to markdown."""
        try:
            block_rect = block["bbox"]
            if block_rect[1] < 50 or block_rect[3] > page_height - 50:
                return ""  # Skip headers and footers

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

                    if self.is_horizontal_line(text):
                        line_text += "\n---\n"
                        continue
                    
                    text = self.clean_text(text)

                    if text.strip():
                        header_level = self.get_header_level(font_size)                   
                        if header_level > 0:
                            text = f"\n{'#' * header_level} {text}\n\n"

                        else:
                            is_list_item = self.is_bullet_point(text) or self.is_numbered_list_item(text)

                            if is_list_item:
                                marker, content = re.split(r'(?<=^[•◦▪▫●○\d.)])\s*', text, 1)
                                formatted_content = self.apply_formatting(content, flags)
                                text = f"{marker} {formatted_content}"
                            else:
                                text = self.apply_formatting(text, flags)
                    
                    for link in links:
                        if fitz.Rect(span_rect).intersects(link["rect"]):
                            text = f"[{text.strip()}]({link['uri']})"
                            break

                    line_text += text

                if last_y1 is not None:
                    avg_last_font_size = sum(last_font_size) / len(last_font_size) if last_font_size else 0
                    avg_current_font_size = sum(curr_font_size) / len(curr_font_size)
                    font_size_changed = abs(avg_current_font_size - avg_last_font_size) > 1

                    if abs(line["bbox"][3] - last_y1) > 2 or font_size_changed:
                        block_text += "\n"
                
                block_text += self.clean_text(line_text) + " "
                last_font_size = curr_font_size
                last_y1 = line["bbox"][3]

            markdown_content = ""
            lines = block_text.split('\n')
            for i, line in enumerate(lines):
                clean_line = self.clean_text(line)

                if not in_code_block:
                    code_lang = self.detect_code_block(prev_line, clean_line)
                    if code_lang:
                        in_code_block = True
                        code_block_lang = code_lang
                        code_block_content = prev_line + "\n" + clean_line + "\n"
                        prev_line = clean_line
                        continue

                if in_code_block:
                    code_block_content += clean_line + "\n"
                    if i == len(lines) - 1 or self.detect_code_block(clean_line, lines[i+1]) != code_block_lang:
                        markdown_content += f"```{code_block_lang}\n{code_block_content}```\n\n"
                        in_code_block = False
                        code_block_content = ""
                        code_block_lang = None
                else:
                    if self.is_bullet_point(clean_line):
                        markdown_content += "\n" + self.convert_bullet_to_markdown(clean_line)
                        list_counter = 0
                    elif self.is_numbered_list_item(clean_line):
                        list_counter += 1
                        markdown_content += "\n" + self.convert_numbered_list_to_markdown(clean_line, list_counter)   
                    else:
                        markdown_content += f"{clean_line}\n"
                        list_counter = 0

                prev_line = clean_line

            return markdown_content + "\n"
        except Exception as e:
            self.logger.error(f"Error processing text block: {e}")
            self.logger.exception(traceback.format_exc())
            return ""

    def process_image_block(self, page, block):
        """Process an image block and convert it to markdown."""
        try:
            image_rect = block["bbox"]
            pix = page.get_pixmap(clip=image_rect)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            image_filename = f"outputs/{self.pdf_filename}_image_{int(page.number)+1}_{block['number']}.png"
            image.save(image_filename)

            caption = self.caption_image(image)

            return f"![{caption}]({image_filename})\n\n"
        except Exception as e:
            self.logger.error(f"Error processing image block: {e}")
            self.logger.exception(traceback.format_exc())
            return ""

    def get_header_level(self, font_size):
        """Determine header level based on font size."""
        if font_size > 24:
            return 1
        elif font_size > 20:
            return 2
        elif font_size > 18:
            return 3
        elif font_size > 16:
            return 4
        elif font_size > 14:
            return 5
        elif font_size > 12:
            return 6
        else:
            return 0

    def post_process_markdown(self, markdown_content):
        """Post-process the markdown content."""
        try:
            markdown_content = re.sub(r'\n{3,}', '\n\n', markdown_content)  # Remove excessive newlines
            markdown_content = re.sub(r'(\d+)\s*\n', '', markdown_content)  # Remove page numbers
            markdown_content = re.sub(r' +', ' ', markdown_content)  # Remove multiple spaces
            markdown_content = re.sub(r'\s*(---\n)+', '\n\n---\n', markdown_content)  # Remove duplicate horizontal lines

            def remove_middle_headers(match):
                line = match.group(0)
                # Keep the initial header and remove all subsequent '#' characters
                return re.sub(r'(^#{1,6}\s).*?(?=\n)', lambda m: m.group(1) + re.sub(r'#', '', m.group(0)[len(m.group(1)):]), line)
            
            markdown_content = re.sub(r'^#{1,6}\s.*\n', remove_middle_headers, markdown_content, flags=re.MULTILINE) # Remove headers in the middle of lines         
            return markdown_content
        except Exception as e:
            self.logger.error(f"Error post-processing markdown: {e}")
            self.logger.exception(traceback.format_exc())
            return markdown_content

    def save_markdown(self, markdown_content):
        """Save the markdown content to a file."""
        try:
            os.makedirs("outputs", exist_ok=True)
            with open(f"outputs/{self.pdf_filename}.md", "w", encoding="utf-8") as f:
                f.write(markdown_content)
            self.logger.info("Markdown content saved successfully.")
        except Exception as e:
            self.logger.error(f"Error saving markdown content: {e}")
            self.logger.exception(traceback.format_exc())

def main():
    parser = argparse.ArgumentParser(description="Extract markdown-formatted content from a PDF file.")
    parser.add_argument("pdf_path", help="Path to the input PDF file")
    args = parser.parse_args()

    extractor = MarkdownPDFExtractor(args.pdf_path)
    extractor.extract()

if __name__ == "__main__":
    main()