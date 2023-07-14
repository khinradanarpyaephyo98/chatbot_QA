import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
# Embeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub

documents = []
for file in os.listdir('docs'):
    if file.endswith('.pdf'):
        pdf_path = './docs/' + file
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

print("Number of documents :",len(documents))


text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
documents = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(documents, embeddings)

os.environ["HUGGINGFACEHUB_API_TOKEN"] ="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"
llm=HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":0, "max_length":512})
chain = load_qa_chain(llm, chain_type="stuff")

query=input("Write the query you want to ask :  ")
docs = db.similarity_search(query)
print("Answer :",chain.run(input_documents=docs, question=query))

