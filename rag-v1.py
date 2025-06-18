from langchain_community.llms import Ollama
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.prompts.chat import ChatPromptTemplate
# importing model
llm = Ollama(model = "llama3.2")

#uploading file
loader = TextLoader("Mango.txt")
#load file
data = loader.load()
#split the file into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap = 500)
chunks = splitter.split_documents(data)

#store in vector database
faiss_vectorstore = FAISS.from_documents(
    documents = chunks, 
    embedding = OllamaEmbeddings(model="llama3.2"),
    )
retriever = faiss_vectorstore.as_retriever()



RAG_PROMPT = """
Answer the question based on the context below. If you can't answer the question, reply "I don't know".

Question:
{question}

Context:
{context}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

llm = Ollama(model = "llama3.2")

# chains
parser = StrOutputParser()
def format_docs(data):
    return "\n".join(doc.page_content for doc in data)
rag_chain = ({"context" : retriever | format_docs, "question": RunnablePassthrough()}
             | rag_prompt
             | llm 
             | StrOutputParser()
             )

response = rag_chain.invoke("What is mango?")

print(response)
