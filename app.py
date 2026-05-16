import os
from dotenv import load_dotenv
import streamlit as st

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

# ======================================================
# Load Environment Variables
# ======================================================
load_dotenv()

# Check API Key
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# ======================================================
# Streamlit Configuration
# ======================================================
st.set_page_config(
    page_title="AI PDF Chatbot (RAG)",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI PDF Chatbot (RAG)")
st.write("Upload a PDF and ask questions based on the document content.")

# ======================================================
# Create Required Folders
# ======================================================
os.makedirs("data", exist_ok=True)
os.makedirs("vector_db", exist_ok=True)

# ======================================================
# Sidebar - PDF Upload
# ======================================================
with st.sidebar:
    st.header("📂 Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"]
    )

# ======================================================
# Main Application Logic
# ======================================================
if uploaded_file is not None:
    # Save uploaded PDF
    file_path = os.path.join("data", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"PDF uploaded successfully: {uploaded_file.name}")

    # --------------------------------------------------
    # Process PDF
    # --------------------------------------------------
    with st.spinner("Processing PDF... Please wait."):
        # Load PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)

        # PDF Statistics
        st.subheader("📊 PDF Statistics")
        st.write(f"📄 Total Pages: {len(documents)}")
        st.write(f"🧩 Total Chunks: {len(chunks)}")

        # --------------------------------------------------
        # Local Embeddings (No API Required)
        # --------------------------------------------------
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # --------------------------------------------------
        # Create Chroma Vector Database
        # --------------------------------------------------
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="vector_db"
        )

        # Sample Vector Preview
        if chunks:
            sample_embedding = embeddings.embed_query(
                chunks[0].page_content[:1000]
            )

            st.subheader("🔢 Vector Preview")
            st.write("Vector Dimension:", len(sample_embedding))
            st.write("First 10 Values:", sample_embedding[:10])

        # --------------------------------------------------
        # Gemini LLM for Answer Generation
        # --------------------------------------------------
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0
        )

    # ==================================================
    # Ask Question
    # ==================================================
    question = st.text_input(
        "💬 Ask a question about the PDF:"
    )

    if question:
        # Retrieve Relevant Chunks
        with st.spinner("Searching relevant content..."):
            retriever = vector_db.as_retriever(
                search_kwargs={"k": 3}
            )
            docs = retriever.invoke(question)

            # Build Context
            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            # Prompt
            prompt = f"""
You are a helpful AI assistant.

Answer the user's question ONLY using the provided PDF context.
If the answer is not present in the context, reply exactly:
I could not find the answer in the PDF.

PDF Context:
{context}

Question:
{question}
"""

        # Generate Answer
        with st.spinner("Generating answer..."):
            response = llm.invoke(prompt)

        # Show Answer
        st.subheader("🤖 Answer")
        st.write(response.content)

        # --------------------------------------------------
        # Retrieved Chunks
        # --------------------------------------------------
        with st.expander("📚 Retrieved Chunks"):
            for i, doc in enumerate(docs, 1):
                st.markdown(f"### Chunk {i}")
                st.write(doc.page_content[:1000])

        # --------------------------------------------------
        # Stored Vector Data
        # --------------------------------------------------
        with st.expander("🗄️ View Stored Vector Database Data"):
            data = vector_db.get()

            st.write("Total Stored Documents:", len(data["documents"]))

            for i, doc in enumerate(data["documents"][:5], 1):
                st.markdown(f"### Stored Document {i}")
                st.write(doc[:1000])

            if chunks:
                st.markdown("### Sample Embedding Values (First 20)")
                st.write(sample_embedding[:20])

else:
    st.info("Please upload a PDF file to get started.")