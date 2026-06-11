from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

import streamlit as st
import tempfile


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="PDF Chatbot",
    layout="centered"
)

st.title("PDF Chatbot")
st.write("""Upload a PDF and ask questions about its content.\n"""
"""Document size should be less than 2MB.""")


# -----------------------------
# LOAD API KEY FROM SECRETS
# -----------------------------
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    st.error(
        "GOOGLE_API_KEY not found in Streamlit Secrets."
    )
    st.stop()


# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file:

    try:

        # Save PDF temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(uploaded_file.getvalue())
            pdf_path = temp_file.name

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        st.write(f"Pages Loaded: {len(documents)}")

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

        chunks = splitter.split_documents(documents)



        # Handle empty PDFs
        if len(chunks) == 0:
            st.error(
                "No text could be extracted from the PDF."
            )
            st.stop()

        # Create embedding model
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=API_KEY
        )

        # Test embedding API
        test_vector = embeddings.embed_query(
            "hello world"
        )


        # Create vector store
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )


        # Create Gemini model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=API_KEY
        )

        # Ask Question
        question = st.text_input(
            "Ask a question about the PDF"
        )

        if question:

            retrieved_docs = vector_store.similarity_search(
                question,
                k=min(5, len(chunks))
            )

            context = "\n\n".join(
                doc.page_content
                for doc in retrieved_docs
            )

            prompt = f"""
You are a PDF assistant.

Answer ONLY using the information in the context.

Rules:
1. Do not use outside knowledge.
2.when answering about experience dont tell about interships
2. If the answer is not present in the document, reply:
   "The information is not available in the uploaded document."
3. Be concise and accurate.

Context:
{context}

Question:
{question}
"""

            response = llm.invoke(prompt)

            st.subheader("Answer")
            st.write(response.content)

            with st.expander(
                "View Retrieved Chunks"
            ):
                for i, doc in enumerate(
                    retrieved_docs,
                    start=1
                ):
                    st.markdown(
                        f"### Chunk {i}"
                    )
                    st.write(
                        doc.page_content
                    )
                    st.divider()

    except Exception as e:

        st.error(
            f"Application Error: {str(e)}"
        )
