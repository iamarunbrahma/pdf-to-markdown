# PDF to Markdown Extractor

## Table of Contents

- [Objective](#objective)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Performance and Accuracy](#performance-and-accuracy)
- [Limitations](#limitations)
- [Use in Downstream Tasks](#use-in-downstream-tasks)
- [Contributing](#contributing)
- [License](#license)

## Objective

This project aims to extract markdown-formatted content from PDF files, specifically designed for downstream tasks such as Retrieval Augmented Generation (RAG). It preserves various markdown elements such as tables, images, links, bold and italic text, blockquotes, code blocks, and other markdown-specific syntax. The script utilizes Python libraries like PyMuPDF (fitz), pdfplumber, pytesseract, and others to achieve accurate extraction and conversion, focusing solely on converting PDF files to Markdown format.

## Features

- Extracts text, images, tables, and code blocks from PDF files
- Converts PDF content to markdown format optimized for RAG and other NLP tasks
- Preserves formatting for bold, italic, tables, images, links, lists, and code blocks
- Handles complex layouts including multi-column text
- Performs OCR on images to extract text
- Generates image captions using a pre-trained model
- Outputs clean, structured markdown suitable for information retrieval and text generation tasks

## Requirements

- Python 3.8+
- PyMuPDF (fitz)
- pdfplumber
- pytesseract
- OpenCV (cv2)
- numpy
- Pillow (PIL)
- transformers
- torch

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/iamarunbrahma/pdf-to-markdown.git
   cd pdf-to-markdown
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Install Tesseract OCR:
   - On Ubuntu: `sudo apt-get install tesseract-ocr`
   - On macOS: `brew install tesseract`
   - On Windows: Download and install from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

## Usage

Run the script with the path to your PDF file as an argument:

```
python extract.py --pdf_path path/to/your/file.pdf
```

The extracted markdown content will be saved in the `outputs` directory with the same name as the input PDF file, but with a `.md` extension.

## Performance and Accuracy

The script is designed to handle various PDF layouts and content types, with a focus on producing high-quality markdown for downstream NLP tasks:

- **Accuracy**: The extractor aims for high accuracy in preserving the original document's structure and formatting. It handles common elements like text, tables, images, links, and code blocks well, ensuring the output is suitable for tasks like RAG. However, very complex layouts or PDFs with non-standard formatting might require manual review.

- **Speed**: The processing time depends on the PDF's size and complexity. On average, for a 10-page PDF with mixed content (text, images, tables, and code blocks), the extraction process typically takes about 30-60 seconds on a modern computer.

- **Optimization for RAG**: The output is structured to facilitate easy parsing and chunking for RAG systems, with clear delineation between different sections and content types.

## Limitations

- This tool is specifically designed for PDF to Markdown conversion and does not handle other file formats.
- Very large PDFs (100+ pages) may require significant processing time.
- PDFs with complex mathematical formulas or specialized symbols may not be perfectly converted.
- Scanned PDFs without embedded text will rely on OCR, which may not be 100% accurate.

## Use in Downstream Tasks

The markdown output from this extractor is particularly well-suited for:

1. **Retrieval Augmented Generation (RAG)**: The structured markdown can be easily indexed and retrieved, providing context for language models in RAG systems.
2. **Text Summarization**: Clean, well-formatted markdown facilitates more accurate summarization of document content.
3. **Information Extraction**: The preserved structure aids in extracting specific information from documents.

## Contributing

Contributions to improve the extractor's accuracy, speed, or feature set are welcome, especially those that enhance its utility for RAG and other NLP tasks. Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.