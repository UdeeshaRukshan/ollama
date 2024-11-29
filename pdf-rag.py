#1. Ingest PDF Files
#2. Extract text from PDF files and split into small chunks
#3. Send the chunks to the embedding model
#4. Save the embeddings to a vector database
#5. Perform similarity search on the vector database to find similar chunks
#6. Retrieve the similar documents and present them to the user

## run pup install -r requirements.txt

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader

doc_path = "./data/BOI.pdf"
model = "llama3.2"

if doc_path:
    loader  = UnstructuredPDFLoader(file_path=doc_path)
    data = loader.load()
    print("Done loading...")
else:
    print("Upload a PDF file to continue...") 


#Preview first page
content = data[0].page_content
print(content[:100])

#End of PDF Ingestion

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Split and chunk

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
chunks =text_splitter.split_documents(data)
print("Done splitting...")

# Add to vector database
import ollama
ollama.pull("nomic-embed-text")



vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="simple-rag",
)

print("done adding to vector database...")

## === Retrieval ===

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama

from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

llm = ChatOllama(model="llama3.2")

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
You are an Ai language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant document from a vector database.
 By generating multiple perspectives on the  on the user question ,your goal is to help the user overcome some of the limitations of the distance-based similarity search. 
 Provide these alternative questions separated by newlines Original question: {question}"""
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(),llm, prompt=QUERY_PROMPT
    )
# RAG prompt
template = """ Answer the question based ONLY on the following context:
{context}
Questions: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

#res = chain.invoke(input=("what is the document about?"))

#res= chain.invoke(input=("what are the main pointes as a business owner I should be aware of?"))

res = chain.invoke(input=("How to report BOI?"))

print(res)