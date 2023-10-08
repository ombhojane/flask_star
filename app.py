from flask import Flask, render_template, request
import os
import fitz
import openai
import tempfile
import concurrent.futures
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator

app = Flask(__name__)

# Set your OpenAI API key here
openai.api_key = "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"

# Define routes and views here
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle file upload
        uploaded_file = request.files["pdf_file"]

        if uploaded_file.filename != "":
            # Save the uploaded PDF to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            # Process and analyze the uploaded PDF
            text = ""
            with fitz.open(temp_file_path) as doc:
                for page_num in range(5, len(doc)):
                    page = doc[page_num]
                    text += page.get_text()

            # Initialize the text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=4000,
                chunk_overlap=200,
                length_function=len
            )

            # Split the text into chunks
            chunks = text_splitter.split_text(text=text)

            # Initialize an empty list to store GPT-3 responses for each chunk
            responses = []

            # Use concurrent.futures to process chunks in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_chunk, chunk) for chunk in chunks]

                # Wait for all futures to complete
                concurrent.futures.wait(futures)

                # Get the results
                responses = [future.result() for future in futures]

            # Display the GPT-3 responses
            return render_template("results.html", responses=responses)

    return render_template("index.html")

# Function to process a text chunk using the OpenAI GPT-3.5 Turbo model
def process_chunk(chunk):
    # Control the rate of requests to stay within the limit (3 requests per minute)
    time.sleep(21)  # Adjust the sleep time as needed

    # Generate GPT-3 response for the current chunk
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use the GPT-3.5 Turbo model
        messages=[
            {"role": "system", "content": "You are going to identify the inconsistencies, errors, and omissions, and provide effective recommendations to improve this text."},
            {"role": "user", "content": chunk}
        ]
    )

    # Extract the response text
    return response.choices[0].message["content"]

if __name__ == "__main__":
    app.run(debug=True)
