import streamlit as st
import os
from groq import Groq
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import  HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()
Groq(api_key=os.environ.get("GROQ_API_KEY"))

# importing model
from langchain_groq import ChatGroq
llm = ChatGroq(
    groq_api_key = os.environ.get("GROQ_API_KEY"),

    model = "llama-3.3-70b-versatile"
)

def main():
    st.title('RAG Q&A System')
#  upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type = 'pdf')

    # st.write(pdf)
    if pdf is not None:
        with st.spinner("Running..."):
            try:
                pdf_reader = PdfReader(pdf)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
            
                # Split text into chunks

                splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap = 100)
                chunks = splitter.split_text(text=text)
                #st.write(chunks)

                #embeddings and store in vector store
                try:
                    faiss_vectorstore = FAISS.from_texts(
                    chunks,HuggingFaceEmbeddings(model_name = "all-MiniLm-L6-v2"),
                    )
                    faiss_vectorstore.save_local("faiss_store")
                except Exception as e:
                    st.error(f"Error in vector store: {e}")
                    return
                
                #Ask query
                try:
                    retriever = faiss_vectorstore.as_retriever()
                    qa = RetrievalQA.from_chain_type(llm, retriever = retriever)
                    query = st.text_input("Ask question about your PDF file:")
                    if st.button("Submit"):
                        if query != "":
                            result = qa.run(query)
                            st.subheader("Answer:")
                            st.info(result)
                            
                        else:
                            st.error("Please enter a question.")  
                            
                except Exception as e:
                    st.error(f"Error in Query and Answer: {e}")

            except Exception as e:
                    st.error(f"Error in processing PDF file: {e}")
    else:  
        st.error("Please upload a PDF file.")

if __name__== '__main__':
    main()




































        
