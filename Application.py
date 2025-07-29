import logging
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch
import re
import nltk
import pdfplumber  # Ensure this is imported
import fitz  # PyMuPDF for highlighting
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import chainlit as cl

# Initialize logger
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

# Initialize models
bi_encoder = SentenceTransformer('multi-qa-mpnet-base-dot-v1')  # Bi-encoder for initial retrieval
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-large")  # Switch to flan-t5-large for better accuracy

# Global variables to store processed data
text_chunks = []
embeddings = []

# Function to extract text line by line from the PDF, preserving line spacing and relative positioning
def extract_text_line_by_line(pdf_path):
    extracted_lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            previous_y = None
            page_lines = []
            for line in page.extract_text(x_tolerance=1, y_tolerance=1).splitlines():
                current_y = page.extract_words()[0]['top'] if page.extract_words() else None
                if previous_y is not None and current_y is not None:
                    line_spacing = current_y - previous_y
                    if line_spacing > 15:  # Threshold for significant spacing
                        page_lines.append("")  # Add an empty line for spacing
                page_lines.append(line)
                previous_y = current_y
            extracted_lines.extend(page_lines)
            extracted_lines.append("")  # Add an empty line between pages
    return extracted_lines

# Function to clean up the extracted text
def clean_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

# Function to split text into manageable chunks using NLTK
def chunk_text(text, max_chunk_size=750):  # Increased chunk size to improve context retrieval
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    logging.debug(f"Text chunks created: {chunks[:3]}")  # Log the first 3 chunks
    return chunks

# Function to generate embeddings for the text chunks
def generate_embeddings(text_chunks):
    embeddings = bi_encoder.encode(text_chunks)
    logging.debug(f"Generated embeddings: {embeddings[:3]}" if len(embeddings) > 0 else "No embeddings generated")
    return np.array(embeddings)

# Function to retrieve top N most relevant chunks and concatenate them
def retrieve_top_n_chunks(query, text_chunks, n=3, max_length=1000):
    query_embedding = bi_encoder.encode([query])
    similarities = cosine_similarity(query_embedding, generate_embeddings(text_chunks))[0]
    top_n_indices = np.argsort(similarities)[-n:][::-1]
    top_chunks = [text_chunks[i] for i in top_n_indices]
    combined_text = " ".join(top_chunks)
    if len(combined_text) > max_length:
        combined_text = combined_text[:max_length]
    logging.debug(f"Combined top chunks: {combined_text}")
    return combined_text

# Main function to find the most relevant answer using top-N retrieval + QA
def find_precise_answer(query, text_chunks):
    query_normalized = re.sub(r'[^a-zA-Z0-9 ]', '', query).lower().strip()
    combined_text = retrieve_top_n_chunks(query, text_chunks, n=3)
    lines = combined_text.splitlines()
    best_match = None
    max_similarity = 0
    for line in lines:
        if ":" in line:
            key, value = map(str.strip, line.split(":", 1))
            key_normalized = re.sub(r'[^a-zA-Z0-9 ]', '', key).lower().strip()
            similarity = cosine_similarity(
                bi_encoder.encode([query_normalized]),
                bi_encoder.encode([key_normalized])
            )[0][0]
            if similarity > max_similarity and similarity > 0.7:  # Threshold for similarity
                max_similarity = similarity
                best_match = value.strip()
    if best_match:
        return best_match
    response = qa_pipeline(f"Context: {combined_text}\nQuestion: {query}\nAnswer:", max_length=100)
    return response[0]['generated_text']

# Function to highlight relevant sections in the PDF
def highlight_relevant_sections(pdf_path, relevant_chunks, output_path):
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        for chunk in relevant_chunks:
            text_instances = page.search_for(chunk)
            for inst in text_instances:
                page.add_highlight_annot(inst)
    doc.save(output_path)
    logging.info(f"Highlighted PDF saved to {output_path}")

@cl.on_chat_start
async def start():
    """Prompts the user to upload multiple PDFs and allows multiple questions."""
    global text_chunks, embeddings  # Declare global to modify these variables

    all_text = ""  # To store text from all PDFs

    # Step 1: Ask the user to upload a PDF file (one at a time)
    while True:
        pdf_file = (await cl.AskFileMessage(
            content="Please upload a PDF file. Type 'Done' to stop uploading PDFs!\n(Please wait if you have already uploaded a PDF...)",
            accept=["application/pdf"],  # Only allow PDF files
            max_size_mb=5  # Optional: Set size limit (e.g., 5MB)
        ).send())[0]

        if pdf_file:
            logging.debug(f"User uploaded: {pdf_file.path}")
            extracted_lines = extract_text_line_by_line(pdf_file.path)
            pdf_text = "\n".join(extracted_lines)  # Join lines to create the final text
            logging.debug(f"Extracted Text: {pdf_text[:500]}")  # Log the first 500 characters of the extracted text
            all_text += pdf_text + "\n\n"  # Combine text from all PDFs

        # Ask if they want to upload another PDF
        user_message = await cl.AskUserMessage(content="Do you want to upload another PDF? (Type Yes else No)").send()
        if user_message['output'].lower() == 'no':
            break

    if not all_text.strip():
        await cl.Message(content="The uploaded PDFs are empty. Please upload valid PDFs.").send()
        return

    # Step 2: Display the extracted text (all the content from the PDFs)
    await cl.Message(content=f"**Extracted content from the PDFs:**\n\n{all_text}...").send()

    # Step 3: Chunk the combined extracted text
    text_chunks = chunk_text(all_text)

    # Step 4: Enter a loop to handle multiple questions
    while True:
        user_message = await cl.AskUserMessage(content="What question would you like to ask about the document? (Please Type 'End' to exit!)").send()
        query = user_message['output'].strip()  # Access content from the dictionary
        logging.debug(f"User question: {query}")
        if query.lower() == 'end':
            await cl.Message(content="Thank you for using the app. Have a nice day!").send()
            break
        if not query:
            await cl.Message(content="Please enter a valid question.").send()
            continue
        precise_answer = find_precise_answer(query, text_chunks)
        combined_text = retrieve_top_n_chunks(query, text_chunks, n=3)
        highlighted_chunk = re.sub(
            re.escape(precise_answer),
            f"**{precise_answer}**",
            combined_text,
            flags=re.IGNORECASE
        )
        await cl.Message(
            content=(
                f"**Answer:**\n\n{precise_answer}\n\n"
                f"**Extracted Answer From The Following Chunk:**\n\n{highlighted_chunk}\n\n"
            )
        ).send()










































































































