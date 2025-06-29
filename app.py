# ========= Importing Required Libraries ============
import streamlit as st
from dotenv import load_dotenv
import os
from io import BytesIO
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# LangChain and OpenAI-related imports
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

# ====== Streamlit page setup with title, icon, and layout ======
st.set_page_config(
    page_title="InsightMate",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ Utility Function to Generate PDF of Q&A ============
def generate_pdf(question, answer, history):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    textobject = c.beginText(40, height - 40)
    textobject.setFont("Helvetica", 11)

    # Write question, answer, and history line by line
    lines = [
        f"Question: {question}",
        "",
        f"Answer: {answer}",
        "",
        "---",
        "Conversation History:",
        history
    ]

    for line in lines:
        for subline in line.split('\n'):
            textobject.textLine(subline)

    c.drawText(textobject)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ============ Load Environment Variables ============
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# ============ Hardcoded User Login System ============
users = {
    "manager": {"password": "manager123", "role": "Manager"},
    "hr": {"password": "hr123", "role": "HR"},
    "intern": {"password": "intern123", "role": "Intern"}
}

# Initialize session state if not already set
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None

# ============ Sidebar UI for Dark Mode and Logout ============
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    dark_mode = st.checkbox("🌗 Dark Mode", value=False)

    st.markdown("---")
    st.markdown(f"👤 Logged in as **{st.session_state.role}**")
    if st.button("🔒 Logout"):
        st.session_state.logged_in = False
        st.session_state.role = None
        st.rerun()

# Apply custom CSS styling for dark mode
if dark_mode:
    st.markdown(
        """
        <style>
        /* Custom styles for dark mode (backgrounds, text, input fields, etc.) */
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #0e1117 !important; 
            color: white !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #161b22 !important;
            color: white !important;
        }
        h1, h2, h3, h4, h5, h6, p, span, div, label {
            color: white !important;
        }
        input, .stTextInput>div>div>input {
            background-color: #1c1f26 !important;
            color: white !important;
            border: 1px solid #333 !important;
        }
        button, .stButton>button {
            background-color: #30363d !important;
            color: white !important;
            border: none !important;
        }
        [data-testid="stFileUploader"] {
            background-color: #1c1f26 !important;
            color: white !important;
            border: 1px solid #333 !important;
            border-radius: 8px !important;
        }
        [data-testid="stFileUploader"] div div span {
            color: #000000 !important;
        }
        [data-testid="stFileUploader"] svg {
            fill: #000000 !important;
        }
        [data-testid="stFileUploader"] label {
            color: white !important;
        }
        .stDownloadButton button {
            background-color: #30363d !important;
            color: white !important;
        }
        .markdown-text-container, .stMarkdown, .stText {
            color: white !important;
        }
        .stExpanderHeader {
            background-color: #161b22 !important;
            color: white !important;
        }
        hr {
            border-top: 1px solid #444 !important;
        }
        [data-testid="stTextArea"] textarea {
            background-color: #0e1117 !important;
            color: white !important;
            border: 1px solid #333 !important;
        }
        [data-testid="stTextArea"] {
            background-color: #0e1117 !important;
        }

        /* Hover effect for buttons in dark mode */
        button:hover, .stButton>button:hover, .stDownloadButton button:hover {
            background-color: #2d6cdf !important;
            color: white !important;
            border: 1px solid #2d6cdf !important;
            transition: 0.2s ease;
            cursor: pointer;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

# ============ Login Interface ============
if not st.session_state.logged_in:
    st.title("🔐 InsightMate Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.role = users[username]["role"]
            st.rerun()
        else:
            st.error("Invalid username or password")
    st.stop()

# ============ File Upload Section ============
if not openai_key:
    st.error("OPENAI_API_KEY not found in .env")
    st.stop()

st.title("💼 InsightMate – Enterprise Knowledge Assistant")

uploaded_files = st.file_uploader(
    "📄 Upload internal documents (PDF or TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

docs = []

# Load files using appropriate loader
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file.flush()
                loader = PyMuPDFLoader(tmp_file.name)
                docs.extend(loader.load())
        elif uploaded_file.type == "text/plain":
            txt_content = uploaded_file.read().decode("utf-8")
            docs.append(Document(page_content=txt_content))
    st.success(f"✅ Loaded {len(docs)} documents.")

# ============ Create Embeddings & FAISS Vector Store ============
if docs:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    st.info(f"🧩 Split into {len(chunks)} chunks.")

    if len(chunks) == 0:
        st.warning("⚠️ No content to process. Please upload valid documents.")
        st.stop()

    with st.spinner("🔄 Generating embeddings..."):
        os.environ.pop("SSL_CERT_FILE", None)
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(chunks, embeddings)
    st.success("✅ FAISS vector store created.")

# ============ Main QA Chat Section ============
if docs:
    st.markdown("---")
    st.subheader("💬 Ask a question")

    role = st.session_state.role
    question = st.text_input("🔍 Enter your question")

    # Prompt template with role-based filtering
    prompt_template = """
    You are a helpful assistant for enterprise teams.
    Respond to the following question using ONLY the information from the context provided below.

    Only show information that someone in the role of a "{role}" is allowed to access.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(
        input_variables=["context", "question", "role"],
        template=prompt_template
    )

    llm = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo")

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question"
        )

    llm_chain = LLMChain(llm=llm, prompt=prompt, memory=st.session_state.memory)

    if st.button("🚀 Get Answer") and question:
        # Perform similarity search
        docs = vector_store.similarity_search(question, k=4)

        # Role-based keyword filter
        allowed_keywords = {
            "Manager": [],
            "HR": ["compliance", "review", "policy", "documentation", "employee", "leave"],
            "Intern": ["project", "status", "update", "milestone", "timeline", "meeting"]
        }

        question_lower = question.lower()
        filtered_docs = []

        for doc in docs:
            content = doc.page_content.lower()

            if role == "Manager":
                filtered_docs.append(doc)
            else:
                if any(kw in question_lower for kw in allowed_keywords[role]) and \
                   any(kw in content for kw in allowed_keywords[role]):
                    filtered_docs.append(doc)

        if not filtered_docs:
            st.warning("⚠️ No information available for your role based on this question.")
            st.stop()

        context = "\n\n".join(doc.page_content for doc in filtered_docs)

        # Run final LLM response
        response = llm_chain.run({
            "context": context,
            "question": question,
            "role": role
        })

        st.markdown("### 🧠 Answer:")
        st.success(response)

        # Display conversation memory
        with st.expander("🗂️ Conversation Memory"):
            formatted_memory = st.session_state.memory.buffer.replace("\n", "<br>")
            if dark_mode:
                st.markdown(
                    f"""
                    <div style="background-color: #0e1117; color: white; padding: 12px; border: 1px solid #333; border-radius: 6px;">
                    {formatted_memory}
                    </div>
                    """,
                    unsafe_allow_html=True
                )   
            else:
                st.markdown(formatted_memory, unsafe_allow_html=True)

        # Export options
        export_text = f"Question: {question}\n\nAnswer: {response}\n\n---\nConversation History:\n{st.session_state.memory.buffer}"
        st.download_button("📥 Download as TXT", export_text, "insightmate_response.txt", "text/plain")

        pdf_buffer = generate_pdf(question, response, st.session_state.memory.buffer)
        st.download_button("📄 Download as PDF", pdf_buffer, "insightmate_response.pdf", "application/pdf")
