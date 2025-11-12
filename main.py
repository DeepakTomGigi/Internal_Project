import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from typing import List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="ABC Policy Helper",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .chat-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sidebar-info {
        padding: 1rem;
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None


@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system (cached to avoid reloading)"""
    try:
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        if not api_key:
            st.error("⚠️ OPENROUTER_API_KEY not found in .env file")
            st.stop()

        # Check if FAISS index exists
        faiss_path = "faiss_index/org_policy"
        if not os.path.exists(faiss_path):
            st.error(f"❌ FAISS index not found at: {faiss_path}")
            st.info("Please create the FAISS index first by running your document ingestion script.")
            st.stop()

        # Load embeddings
        with st.spinner("Loading embeddings model..."):
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

        # Load FAISS vectorstore
        with st.spinner("Loading knowledge base..."):
            vectorstore = FAISS.load_local(
                faiss_path,
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )

        # Configure retriever
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 20,
                "lambda_mult": 0.5
            }
        )

        # Setup LLM
        llm = ChatOpenAI(
            model="meta-llama/llama-3.3-8b-instruct:free",
            temperature=0.7,
            api_key=api_key,
            base_url=base_url,
            timeout=30,
            max_retries=2
        )

        # ✅ FIXED: Enhanced prompt template with proper variables
        prompt = ChatPromptTemplate.from_template("""You are an HR policy assistant for ABC Organization.

Context from policy documents:
{context}

Question: {question}

Rules:
- Answer ONLY based on the provided context above
- If the answer is not in the context, say "I don't have enough information in the policy documents to answer this question"
- Cite the specific policy section or document when possible
- Use simple, employee-friendly language
- Be concise but complete

Answer:""")

        # Format documents function with improved formatting
        def format_docs(docs: List) -> str:
            if not docs:
                return "No specific context found in the knowledge base."

            formatted = []
            for i, doc in enumerate(docs):
                # Include metadata if available
                metadata_str = ""
                if hasattr(doc, 'metadata') and doc.metadata:
                    source = doc.metadata.get('source', 'Unknown')
                    page = doc.metadata.get('page', 'N/A')
                    metadata_str = f" [Source: {source}, Page: {page}]"

                formatted.append(f"Document {i + 1}{metadata_str}:\n{doc.page_content}")

            return "\n\n---\n\n".join(formatted)

        # Build LCEL chain
        rag_chain = (
                RunnableParallel({
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough()
                })
                | prompt
                | llm
                | StrOutputParser()
        )

        logger.info("✅ RAG system initialized successfully")
        return rag_chain, retriever, vectorstore

    except FileNotFoundError as e:
        st.error(f"❌ File not found: {str(e)}")
        logger.error(f"File not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error initializing RAG system: {str(e)}")
        logger.error(f"Initialization error: {e}", exc_info=True)
        st.stop()


def get_answer(question: str) -> str:
    """Get answer from RAG system"""
    try:
        if st.session_state.rag_chain is None:
            logger.error("RAG chain is None")
            return "Sorry, the system is not initialized properly."

        # Log the question
        logger.info(f"Processing question: {question}")

        # Get answer
        answer = st.session_state.rag_chain.invoke(question)

        # Log success
        logger.info(f"Answer generated successfully (length: {len(answer)})")

        return answer

    except Exception as e:
        logger.error(f"Error getting answer: {e}", exc_info=True)
        return f"❌ Sorry, I encountered an error while processing your question: {str(e)}"


def clear_chat():
    """Clear chat history"""
    st.session_state.messages = []
    st.rerun()


# Main UI
def main():
    # Header
    st.markdown('<div class="chat-header">📚 ABC Policy Helpline</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("ℹ️ About")
        st.markdown("""
        <div class="sidebar-info">
        <b>ABC ORG Policy Chatbot</b><br><br>
        Ask questions about the organization's policies, procedures, and guidelines.
        <br><br>
        <b>Powered by:</b>
        <ul>
        <li>🤖 Llama 3.3 (8B)</li>
        <li>📚 FAISS Vector Database</li>
        <li>🔍 Intelligent Retrieval (MMR)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Sample questions
        st.subheader("💡 Sample Questions")
        sample_questions = [
            "What is the leave policy?",
            "How do I submit a reimbursement?",
            "What are the working hours?",
            "What is the remote work policy?"
        ]

        for q in sample_questions:
            if st.button(q, key=f"sample_{q}", use_container_width=True):
                # Add question to chat
                st.session_state.messages.append({"role": "user", "content": q})
                with st.spinner("Thinking..."):
                    answer = get_answer(q)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                st.rerun()

        st.divider()

        # Settings
        st.subheader("⚙️ Settings")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            clear_chat()

        # Statistics
        if st.session_state.messages:
            st.divider()
            st.subheader("📊 Session Stats")
            user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
            st.metric("Questions Asked", user_msgs)

    # Initialize RAG system on first run
    if st.session_state.rag_chain is None:
        with st.spinner("🚀 Initializing Policy Helper... This may take a minute..."):
            rag_chain, retriever, vectorstore = initialize_rag_system()
            st.session_state.rag_chain = rag_chain
            st.session_state.retriever = retriever
            st.session_state.vectorstore = vectorstore
        st.success("✅ Policy Helper is ready! Ask your first question below.", icon="✅")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me about ABC's policies..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = get_answer(prompt)
                st.markdown(answer)

        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": answer})
        # ✅ FIXED: Removed unnecessary rerun

    # Welcome message if no messages
    if not st.session_state.messages:
        st.info(
            "👋 **Welcome to ABC Policy Helper!**\n\n"
            "Ask me anything about ABC organization's policies. "
            "Try one of the sample questions in the sidebar, or type your own question below.",
            icon="ℹ️"
        )


if __name__ == "__main__":
    main()