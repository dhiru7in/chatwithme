from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import streamlit as st
import tempfile


API_KEY = st.secrets["GOOGLE_API_KEY"]

st.set_page_config(
    page_title="PDF Chatbot with Gemini",
    page_icon="📄",
    layout="centered"
)

st.title("PDF Chatbot with Gemini")
st.write("Upload any PDF file and ask questions about its content.")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        pdf_path = temp_file.name

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    st.write(f"Loaded {len(documents)} pages and created {len(chunks)} text chunks.")

    if not chunks:
        st.error(
            "No text could be extracted from this PDF. "
            "This may happen with scanned documents or image-only PDFs. "
            "Try a different file or use an OCR-enabled loader."
        )
    else:
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=API_KEY)
        vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=API_KEY)
        question = st.text_input("Ask a question about the PDF")

        if question:
            retrieved_docs = vector_store.similarity_search(question, k=8)
            context = "\n\n".join(doc.page_content for doc in retrieved_docs)

            prompt = (
                "You are a helpful assistant. Answer the question using only the information from the provided context. "
                "If the answer is not available in the document, say that the information is not available.\n\n"
                f"Context:\n{context}\n\nQuestion:\n{question}"
            )

            response = llm.invoke(prompt)

            st.subheader("Answer")
            st.write(response.content)

            with st.expander("Retrieved context"):
                for i, doc in enumerate(retrieved_docs, start=1):
                    st.markdown(f"**Chunk {i}**")
                    st.write(doc.page_content)
                    st.divider()

