import os
import time
import json
import shutil
import pdfplumber
import pandas as pd
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv


load_dotenv()
os.environ['NVIDIA_API_KEY'] = os.getenv('NVIDIA_API_KEY')

llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

# Store previous report data for comparison
previous_report = None

# Function to handle PDF upload
def upload_pdf():
    pdf_path = input("Please upload your PDF (provide file path): ")
    if os.path.exists(pdf_path) and pdf_path.endswith(".pdf"):
        destination = os.path.join('./data', os.path.basename(pdf_path))
        shutil.copy(pdf_path, destination)
        print(f"✅ PDF uploaded and saved to: {destination}")
        return destination
    else:
        print("❌ Invalid file path or file type. Please upload a valid PDF.")
        return None

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

        # Save extracted data as JSON
        output_path = pdf_path.replace(".pdf", ".json")
        with open(output_path, "w") as f:
            json.dump(extracted_data, f, indent=4)

        print(f"✅ Extracted and saved structured data from: {pdf_path}")
        return extracted_data
    
    except Exception as e:
        print(f"❌ Error extracting data from {pdf_path}: {str(e)}")
        return None

# Function to create vector embeddings
def vector_embedding():
    try:
        embedding = NVIDIAEmbeddings()

        # Process PDFs from the 'data' folder
        pdf_files = [f"./data/{f}" for f in os.listdir("./data") if f.endswith(".pdf")]
        all_texts = []
        
        for pdf in pdf_files:
            extracted_data = extract_text_and_tables(pdf)
            if extracted_data:
                all_texts.append(extracted_data["text"])  

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        final_document_splitter = text_splitter.create_documents(all_texts)
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

# Function to analyze the report and provide recommendations
def analyze_lab_report(vectors):
    try:
        retriever = vectors.as_retriever()
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        response = retrieval_chain.invoke({'input': "Analyze the lab report and provide lifestyle recommendations."})
        analysis_result = response['answer']
        
        print("\n📋 **Lab Report Analysis:**\n", analysis_result)

        return analysis_result

    except Exception as e:
        print(f"❌ Error in report analysis: {str(e)}")
        return None

# Function to compare two reports
def compare_reports(previous, current):
    try:
        if not previous or not current:
            print("⚠ No previous report to compare.")
            return
        
        prompt_compare = ChatPromptTemplate.from_template(
            """Compare these two medical reports and highlight differences:
            
            🏥 **First Report:** {previous_report}
            📄 **Second Report:** {current_report}
            
            - **Differences in Lab Values**
            - **Changes in Disease Condition**
            - **Improvement or Deterioration**
            """
        )

        retriever = vectors.as_retriever() # type: ignore
        document_chain = create_stuff_documents_chain(llm, prompt_compare)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        response = retrieval_chain.invoke({
            'previous_report': previous,
            'current_report': current
        })
        
        print("\n🔍 **Comparison Between Reports:**\n", response['answer'])

    except Exception as e:
        print(f"❌ Error in report comparison: {str(e)}")

# Main Function
def main():
    global previous_report

    while True:
        pdf_path = upload_pdf()
        if not pdf_path:
            continue
        
        vectors = vector_embedding()
        if not vectors:
            print("❌ Error creating vector store.")
            continue
        
        # Automatically analyze the uploaded PDF
        current_analysis = analyze_lab_report(vectors)

        # Compare if a previous report exists
        if previous_report:
            compare_reports(previous_report, current_analysis)
        
        # Store the current analysis as the previous report
        previous_report = current_analysis

        # Allow Q&A
        ask_question(vectors)

# Function to allow Q&A
def ask_question(vectors):
    while True:
        prompt1 = input("\nEnter your question (or type 'exit' to quit): ")
        if prompt1.lower() == "exit":
            break

        try:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            start = time.process_time()
            response = retrieval_chain.invoke({'input': prompt1})
            print(f"\n⏳ Response time: {time.process_time() - start:.2f} seconds")
            print("\n💡 Answer:", response['answer'])

        except Exception as e:
            print(f"❌ Error retrieving response: {str(e)}")

# Run the program
if __name__ == "__main__":
    main()