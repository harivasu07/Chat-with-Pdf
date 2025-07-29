# PDF Question Answering App 

This interactive app, made using **Chainlit**, allows users to upload multiple files, extract content, and interact with the content. It uses **pdfplumber** to upload the file, and uses **SentenceTransformers** and **flan-t5-large** for the Embedding System and LLM model respectively, thus extracting precise answers from the texts.

## Special Features of the App

- Upload multiple PDFs and extract content.
- Ask questions based on the extracted text.
- Highlight relevant answers in the PDF.
- Allows for multiple question-answering sessions.

## Requirements (Demo Video At the End)

- Python 3.x
- sentence-transformers
- transformers
- fitz (PyMuPDF)
- pdfplumber
- nltk
- chainlit
- scikit-learn
- numpy
- re

## Setup and Usage (Main file : Application.py)

### Clone the Repository (To get all files in your system)

(In Bash)
- `git clone https://github.com/GitHub-arihant/Chat-With_PDF`
- `cd Chat-With_PDF`

### Create Virtual Environment (To isolate the project dependencies)

(For Windows:)
- `python -m venv venv`
- `venv/Scripts/activate`

(For macOS/Linux:)
- `python3 -m venv venv`
- `source venv/bin/activate`

### Install Dependencies (install all the libraries that are used in the Project)

(In Bash)
- `pip install -r required.txt`

### Run the Application

(In Bash)
- `chainlit run Application.py`

### Upload PDFs and Ask Questions:

When prompted, upload your PDFs and ask questions. The app will process the PDFs, extract text, and highlight the answers within the PDFs.

## Demo Video Link

[https://youtu.be/Lp9Tm-dExTo](https://youtu.be/Lp9Tm-dExTo) (Youtube)

or 

Search for keywords: Demo Video For The Chat With Pdf App. Arihant Jain(Channel)
