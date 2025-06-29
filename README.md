# InsightMate – Smart Enterprise Knowledge Assistant

InsightMate is an intelligent enterprise-grade assistant that understands your internal documents (like emails and reports) and generates role-specific answers. Built with LangChain, OpenAI GPT, and FAISS, this project brings Retrieval-Augmented Generation (RAG) to life through a clean Streamlit UI and supports document-based Q&A tailored to user roles such as Manager, HR, and Intern.

## Key Features:-

- Upload Documents – Supports internal documents in PDF or TXT format
- Login-Based Access – Role-based login system (Manager / HR / Intern)
- RAG Pipeline – Uses LangChain to retrieve context-relevant data from documents
- Vector Search – FAISS-powered document chunk retrieval
- OpenAI GPT Integration – LLM-based intelligent responses
- Role-Specific Answers – Each role gets filtered, relevant answers based on predefined keyword policies
- Conversation Memory – Maintains history for multi-turn interactions
- Download Responses – Export answers with conversation memory as .txt or .pdf
- Dark/Light Mode UI – Clean ChatGPT-style theme toggle with responsive design

## Tech Stack & Tools:-

**Category**	     **Stack Used**
Language	         Python 3.10+
LLM	                 OpenAI GPT (via ChatOpenAI)
Frameworks	         LangChain, Streamlit
Search Engine	     FAISS Vector Store
Document Parsing	 PyMuPDF, Plaintext
Memory	             LangChain's ConversationBufferMemory
Utilities	         dotenv, tempfile, reportlab (PDF export)

## Sample Use Case:-

You upload internal company files such as:

- email.txt: Sample internal email about leave policy
- report.pdf: Official leave structure for employees

Then, depending on your role, InsightMate answers differently:

Manager can ask about project delays or client issues
HR can ask about compliance or leave policy
Intern can ask about meeting timelines or casual leave