from dotenv import load_dotenv
import os
import uuid


from dotenv import load_dotenv
import os
import uuid

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
import chromadb

# Load .env
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")


# Embedding model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ChromaDB local storage
client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="documents"
)


# Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY
)


def load_pdf(file_path):

    loader = PyPDFLoader(file_path)

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    for chunk in chunks:

        vector = embedding_model.encode(
            chunk.page_content
        ).tolist()

        unique_id = str(uuid.uuid4())

        collection.add(
            ids=[unique_id],
            documents=[chunk.page_content],
            embeddings=[vector]
        )

    print(f"{len(chunks)} chunks stored")


def ask(question):

    query_embedding = embedding_model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    documents = results.get("documents")

    if not documents or len(documents[0]) == 0:
        return "No relevant information found."

    context = "\n".join(
        documents[0]
    )

    prompt = f"""
Answer ONLY from the provided context.

If the answer is not present, say:
"I could not find this in the documents."

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    return response.content


# Test
if __name__ == "__main__":

    load_pdf("policy.pdf")

    question = input("Ask: ")

    answer = ask(question)

    print("\nAnswer:")
    print(answer)