import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import retrieval_qa as RetrievalQA

# Load environment variables from .env file
load_dotenv()

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please set the OPENAI_API_KEY in a .env file.")

# Step 1: Load the PDF document
document_loader = PyPDFLoader(file_path="NIPS-2017-attention-is-all-you-need-Paper.pdf")
documents = document_loader.load()

# Step 2: Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Number of characters per chunk
    chunk_overlap=200  # Overlap to maintain context
)
doc_splits = text_splitter.split_documents(documents)

# Step 3: Create embeddings and store in FAISS vector store
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(doc_splits, embeddings)

# Step 4: Set up the retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}  # Retrieve top 4 relevant chunks
)

# Step 5: Initialize the ChatOpenAI model
llm = ChatOpenAI(
    model_name="gpt-4o",  # You can use "gpt-3.5-turbo" if preferred
    temperature=0.0       # Low temperature for factual responses
)

# Step 6: Define the prompt template for RAG
prompt_template = """
You are an expert on the "Attention Is All You Need" paper. Use the following context from the paper to answer the question. If the answer isn't in the context, say so and provide a general response based on your knowledge.

**Context:**
{context}

**Question:**
{question}

**Answer:**
"""
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Step 7: Create the RAG chain using RetrievalQA
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # "stuff" combines all retrieved docs into the prompt
    retriever=retriever,
    return_source_documents=True,  # Optional: returns the source chunks
    chain_type_kwargs={"prompt": prompt}
)

# Step 8: Function to query the RAG system
def ask_question(query):
    result = rag_chain.invoke({"query": query})
    print(f"\nQuestion: {query}")
    print(f"Answer: {result['result']}")
    # Optionally print source documents
    # print("\nSources:")
    # for doc in result["source_documents"]:
    #     print(f"- {doc.page_content[:100]}... (Page {doc.metadata['page']})")

# Step 9: Test the RAG system
if __name__ == "__main__":
    try:
        # Example queries
        ask_question("What is the main innovation introduced in the paper?")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
