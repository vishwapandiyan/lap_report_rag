import os
import time
import json
import shutil
import pdfplumber
import pandas as pd
from flask import Flask, request, jsonify
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['NVIDIA_API_KEY'] = os.getenv('NVIDIA_API_KEY')

# Initialize Flask app
app = Flask(__name__)

# Initialize LLM
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

# Store previous report data for comparison
previous_report = None

# Ensure 'data' folder exists
if not os.path.exists("data"):
    os.makedirs("data")

# Function to extract text and tables from PDFs
def extract_text_and_tables(pdf_path):
    extracted_data = {"text": "", "tables": []}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_data["text"] += text + "\n"

                # Extract tables
                tables = page.extract_tables()
                for table in tables:
                    df = pd.DataFrame(table)
                    extracted_data["tables"].append(df.to_dict(orient="records"))

        print(f"✅ Extracted data from: {pdf_path}")
        return extracted_data
    
    except Exception as e:
        print(f"❌ Error extracting data: {str(e)}")
        return None

# Function to create vector embeddings
def vector_embedding(pdf_text):
    try:
        embedding = NVIDIAEmbeddings()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        final_document_splitter = text_splitter.create_documents([pdf_text])
        print(f"✅ Number of document chunks: {len(final_document_splitter)}")

        # Create vector embeddings
        vectors = FAISS.from_documents(final_document_splitter, embedding)
        print("✅ Vector store is ready.")
        return vectors

    except Exception as e:
        print(f"❌ Error during vector embedding: {str(e)}")
        return None

# Chat Prompt Template
prompt = ChatPromptTemplate.from_template(
    """Based on the lab report, extract possible diseases and provide lifestyle modifications.
    
    <context>{context}</context>

    1️⃣ **Predicted Disease/Condition:**  
    2️⃣ **Lifestyle Modifications:**  
      - **Exercise Plan:**  
      - **Hydration Requirements:**  
      - **Diet Schedule:** (Morning, Afternoon, Evening)
    """
)

# API Endpoint: Upload PDF
@app.route("/upload", methods=["POST"])
def upload_pdf():
    global previous_report

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    pdf_file = request.files["file"]
    pdf_path = os.path.join("data", pdf_file.filename)
    pdf_file.save(pdf_path)

    extracted_data = extract_text_and_tables(pdf_path)
    if not extracted_data:
        return jsonify({"error": "Failed to extract data from PDF"}), 500

    vectors = vector_embedding(extracted_data["text"])
    if not vectors:
        return jsonify({"error": "Failed to create vector embeddings"}), 500

    retriever = vectors.as_retriever()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({'input': "Analyze the lab report and provide lifestyle recommendations."})
    current_analysis = response['answer']

    comparison = None
    if previous_report:
        prompt_compare = ChatPromptTemplate.from_template(
            """Compare these two medical reports and highlight differences:
            
            🏥 **First Report:** {previous_report}
            📄 **Second Report:** {current_report}
            
            - **Differences in Lab Values**
            - **Changes in Disease Condition**
            - **Improvement or Deterioration**
            """
        )
        document_chain_compare = create_stuff_documents_chain(llm, prompt_compare)
        retrieval_chain_compare = create_retrieval_chain(retriever, document_chain_compare)

        comparison_response = retrieval_chain_compare.invoke({
            'previous_report': previous_report,
            'current_report': current_analysis
        })
        comparison = comparison_response['answer']

    previous_report = current_analysis

    return jsonify({
        "analysis": current_analysis,
        "comparison": comparison if previous_report else "No previous report to compare"
    })

# API Endpoint: Ask Questions
@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    if "question" not in data:
        return jsonify({"error": "No question provided"}), 400

    prompt1 = data["question"]

    try:
        retriever = vectors.as_retriever() # type: ignore
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        response_time = time.process_time() - start

        return jsonify({
            "question": prompt1,
            "answer": response['answer'],
            "response_time": f"{response_time:.2f} seconds"
        })

    except Exception as e:
        return jsonify({"error": f"Error retrieving response: {str(e)}"}), 500

# Run Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)