"""
PDFPal is an AI chatbot that answers questions related to PDF content based on user prompts.
"""
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_ollama.llms import OllamaLLM

def main():
    """
    Main entry point for the script.
    """
    load_dotenv()
    
    st.set_page_config(page_title="PDFPal", page_icon=":books:")
    st.header("Welcome to PDFPal 😁")
    st.write("PDFPal is an AI chatbot that answers questions related to PDF content based on user prompts.💬")
    
    uploaded_pdf = st.file_uploader(label="Upload your PDF file and ask questions about it.", type="pdf")
    
    if uploaded_pdf is not None:
        pdf_reader = PdfReader(uploaded_pdf)
        pdf_text = ""
        for page in pdf_reader.pages:
            pdf_text += page.extract_text() or ""
        
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        text_chunks = text_splitter.split_text(pdf_text)
        
        embeddings = OllamaEmbeddings(model="llama3")
        knowledge_base = FAISS.from_texts(text_chunks, embeddings)
        
        user_question = st.text_input("Ask a question about the PDF:")
        
        if user_question:
            # Get the relevant documents from the knowledge base
            relevant_docs = knowledge_base.similarity_search(user_question)
            
            # Perform question answering on the relevant documents
            llm = OllamaLLM(model="llama3")
            qa_chain = load_qa_chain(llm, chain_type="stuff")
            answer = qa_chain.run(input_documents=relevant_docs, question=user_question)
            
            # Print the answer
            st.write(answer)

if __name__ == '__main__':
    main()

