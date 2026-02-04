from pathlib import Path
import random
from IPython.display import Markdown, display

from langchain_community.document_loaders import (
    DirectoryLoader, 
    TextLoader, 
    CSVLoader, 
    PyPDFLoader, 
    Docx2txtLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

import tiktoken
from typing import List, Union
from langchain_core.documents import Document

from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from typing import Optional

from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from typing import List, Tuple

from typing import List, Union, Iterable
from langchain_core.documents import Document

import random
import json
from IPython.display import Markdown, display

from typing import Optional, Literal, Dict, Any
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever

from typing import Optional

# 1. Base interfaces (Standardized in langchain-core)
from langchain_core.retrievers import BaseRetriever
from langchain_core.embeddings import Embeddings

# 2. Retrieval Orchestration (Now explicitly in the main langchain package)
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)

# 3. Specialized Package Imports
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
    LongContextReorder,
)

from typing import Optional
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_core.retrievers import BaseRetriever

from typing import Optional, Literal, Dict, Any
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from typing import Optional, Any
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Map extensions to their specific loader classes
LOADER_MAPPING = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": PyPDFLoader,
    ".csv": CSVLoader,
    ".docx": Docx2txtLoader,
}


def langchain_document_loader(tmp_dir: Path):
    documents = []
    
    for ext, loader_cls in LOADER_MAPPING.items():
        # Using DirectoryLoader for each type to handle recursive globbing
        loader = DirectoryLoader(
            str(tmp_dir), 
            glob=f"**/*{ext}", 
            loader_cls=loader_cls,
            # Pass encoding only to loaders that need/support it
            loader_kwargs={"encoding": "utf8"} if ext in [".txt", ".md", ".csv"] else {}
        )
        documents.extend(loader.load())
        
    return documents

def show_random_preview(docs):
    """
    Displays a formatted preview of a random document from the list.
    Handles imports internally to prevent NameErrors.
    """
    # 1. Safety check for empty list
    if not docs:
        print("❌ Error: The document list is empty.")
        return

    # 2. Select a random document
    idx = random.randint(0, len(docs) - 1)
    doc = docs[idx]
    
    # 3. Clean up content: Escape '$' to avoid Markdown math errors
    content_preview = doc.page_content[:1000].replace("$", r"\$").strip()
    
    # 4. Format metadata for readability
    metadata_formatted = json.dumps(doc.metadata, indent=2)
    
    # 5. Build the Markdown string
    md_text = (
        f"## 📄 Document Preview\n"
        f"**Index:** `{idx}` | **Source:** `{doc.metadata.get('source', 'Unknown')}`\n\n"
        f"--- \n"
        f"### **Page Content (First 1000 chars):**\n"
        f"{content_preview}...\n\n"
        f"--- \n"
        f"### **Metadata:**\n"
        f"```json\n"
        f"{metadata_formatted}\n"
        f"```"
    )
    
    # 6. Render it
    display(Markdown(md_text))

def create_text_splitter(
    chunk_size: int = 1600,
    chunk_overlap: int = 200,
    separators: list[str] = None
) -> RecursiveCharacterTextSplitter:
    """
    Create a reusable RecursiveCharacterTextSplitter.
    
    Args:
        chunk_size (int): Maximum number of characters per chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.
        separators (list[str], optional): List of separators in order of priority.
            Default is ["\n\n", "\n", " ", ""].
    
    Returns:
        RecursiveCharacterTextSplitter instance
    """
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]
    
    splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter

def create_advanced_markdown_splitter(chunk_size=1600, chunk_overlap=200):
    # 1. Define the structural boundaries
    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ]
    
    # 2. Initialize the structural splitter
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, 
        strip_headers=False # Keep headers in the text so the LLM sees them
    )
    
    # 3. Initialize your existing recursive splitter for the "fine-tuning"
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    return markdown_splitter, recursive_splitter

def split_documents(documents, splitter: RecursiveCharacterTextSplitter):
    """
    Split a list of documents into chunks using the provided splitter.
    
    Args:
        documents (list): List of LangChain Document objects.
        splitter (RecursiveCharacterTextSplitter): Initialized text splitter.
    
    Returns:
        list: List of text chunks
    """
    chunks = splitter.split_documents(documents)
    print(f"Number of chunks created: {len(chunks)}")
    return chunks

def tiktoken_tokens(
    documents: Union[List[Document], List[str]], 
    model: str = "gpt-3.5-turbo"
) -> List[int]:
    """
    Return the number of tokens for each document using tiktoken (OpenAI tokenizer).
    
    Args:
        documents (List[Document] | List[str]): List of LangChain Document objects or raw text strings.
        model (str): OpenAI model name to determine token encoding (default "gpt-3.5-turbo").
    
    Returns:
        List[int]: Number of tokens per document.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # fallback to default encoding if model is unknown
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens_length = []
    
    for doc in documents:
        if isinstance(doc, Document):
            text = doc.page_content
        elif isinstance(doc, str):
            text = doc
        else:
            raise TypeError("Each item in documents must be a LangChain Document or a string.")
        
        tokens_length.append(len(encoding.encode(text)))
    
    return tokens_length


def select_embeddings_model(
    LLM_service: str = "OpenAI",
    openai_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
    HF_key: Optional[str] = None
):
    """
    Connect to the embeddings API endpoint by specifying the embedding model.

    Args:
        LLM_service (str): Name of the service to use: "OpenAI", "Google", or "HuggingFace".
        openai_api_key (str, optional): API key for OpenAI (required if LLM_service="OpenAI").
        google_api_key (str, optional): API key for Google (required if LLM_service="Google").
        HF_key (str, optional): API key for Hugging Face (required if LLM_service="HuggingFace").

    Returns:
        embeddings object: Initialized embeddings instance for the chosen provider.

    Raises:
        ValueError: If LLM_service is unknown or the required API key is missing.
    """
    
    LLM_service = LLM_service.lower()  # Case-insensitive handling
    
    if LLM_service == "openai":
        if not openai_api_key:
            raise ValueError("OpenAI API key is required.")
        return OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_api_key)
    
    elif LLM_service == "google":
        if not google_api_key:
            raise ValueError("Google API key is required.")
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    
    elif LLM_service == "huggingface":
        if not HF_key:
            raise ValueError("Hugging Face API key is required.")
        return HuggingFaceInferenceAPIEmbeddings(model_name="thenlper/gte-large", api_key=HF_key)
    
    else:
        raise ValueError(f"Unknown LLM_service '{LLM_service}'. Choose from 'OpenAI', 'Google', or 'HuggingFace'.")

LOCAL_VECTOR_STORE_DIR = Path("vectorstores")  # example path, adjust as needed

def create_vectorstore(
    embeddings: Embeddings,
    documents: List[Document],
    vectorstore_name: str,
    vectorstore_dir: str
) -> FAISS:
    """
    Creates a FAISS vectorstore and persists it locally.
    FAISS is Pydantic v2 compatible and significantly faster than Chroma for local use.
    """
    # Define and create the save path
    save_path = Path(vectorstore_dir) / vectorstore_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # FAISS .from_documents is the standard Big Tech entry point for local RAG
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    # Persist to disk (FAISS saves as .faiss and .pkl files)
    vector_store.save_local(str(save_path))
    
    return vector_store

def print_documents(
    docs: Iterable[Union[Document, tuple]],
    search_with_score: bool = False,
    separator: str = "-" * 100
) -> None:
    """
    Helper function to print documents in a readable format.

    Args:
        docs (Iterable[Union[Document, tuple]]): 
            A list of Document objects or tuples (Document, score).
        search_with_score (bool): 
            If True, expects tuples with scores (used in similarity_search_with_score).
        separator (str): 
            String used to separate printed documents.

    Returns:
        None
    """
    output_lines = []

    for i, doc in enumerate(docs):
        if search_with_score:
            # Expecting tuple: (Document, score)
            page_content = doc[0].page_content
            score = doc[-1]  # assuming last element is the score
            output_lines.append(
                f"Document {i+1}:\n\n{page_content}\n\nScore: {score:.3f}\n"
            )
        else:
            # Just a Document object
            output_lines.append(
                f"Document {i+1}:\n\n{doc.page_content}\n"
            )

    print(f"\n{separator}\n".join(output_lines))

def get_vectorstore_retriever(
    vectorstore: VectorStore,
    search_type: Literal["similarity", "mmr", "similarity_score_threshold"] = "similarity",
    k: int = 4,
    score_threshold: Optional[float] = None,
    fetch_k: int = 20
) -> BaseRetriever:
    """Standardizes the instantiation of vectorstore-backed retrievers.

    Args:
        vectorstore: The initialized LangChain VectorStore instance.
        search_type: Algorithm for retrieval. 
            - 'similarity': Standard cosine/L2 distance.
            - 'mmr': Max Marginal Relevance (diversifies results).
            - 'similarity_score_threshold': Filters results by absolute score.
        k: The number of final documents to return to the LLM.
        score_threshold: Minimum relevance score required (range 0.0 to 1.0).
            Only utilized when search_type is 'similarity_score_threshold'.
        fetch_k: Amount of documents to pass to the MMR algorithm for reranking.

    Returns:
        BaseRetriever: A configured retriever object ready for RAG chains.
    """
    
    search_kwargs: Dict[str, Any] = {"k": k}
    
    if search_type == "mmr":
        search_kwargs["fetch_k"] = fetch_k
        
    if score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold

    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )

def create_compression_retriever(
    embeddings: Embeddings,
    base_retriever: BaseRetriever,
    chunk_size: int = 500,
    k: int = 16,
    similarity_threshold: Optional[float] = None,
) -> ContextualCompressionRetriever:
    """
    Factory function to construct a high-precision retrieval pipeline using 
    Contextual Compression.

    The pipeline follows a four-stage transformation process:
    1. Splitting: Breaks docs into granular chunks for precise matching.
    2. De-duplication: Removes semantically redundant information.
    3. Relevance Filtering: Enforces a similarity threshold and limits top-k.
    4. Reordering: Optimizes for the "Lost in the Middle" phenomenon.

    Args:
        embeddings: The embedding model used for redundancy and relevance filtering.
        base_retriever: The primary vector store retriever.
        chunk_size: Target size for internal document splitting.
        k: Maximum number of documents to return after filtering.
        similarity_threshold: Minimum cosine similarity score (0.0 to 1.0).

    Returns:
        A configured ContextualCompressionRetriever instance.
    """
    
    # Initialize transformers with explicit configurations
    transformers = [
        CharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_size // 10,  # Added overlap for context retention
            separator=". "
        ),
        EmbeddingsRedundantFilter(embeddings=embeddings),
        EmbeddingsFilter(
            embeddings=embeddings, 
            k=k, 
            similarity_threshold=similarity_threshold
        ),
        LongContextReorder()
    ]

    # Encapsulate logic in a DocumentCompressorPipeline
    pipeline_compressor = DocumentCompressorPipeline(transformers=transformers)

    return ContextualCompressionRetriever(
        base_compressor=pipeline_compressor,
        base_retriever=base_retriever
    )

def create_cohere_rerank_retriever(
    base_retriever: BaseRetriever, 
    cohere_api_key: str, 
    model: str = "rerank-multilingual-v3.0", 
    top_n: int = 8
) -> ContextualCompressionRetriever:
    """
    Configures a ContextualCompressionRetriever using Cohere's Rerank API.
    
    This factory function enhances a standard retriever by adding a secondary 
    relevance-based scoring pass. It is particularly effective at reducing noise 
    and improving the performance of the downstream LLM.

    Args:
        base_retriever: The initial retriever (e.g., VectorStoreRetriever).
        cohere_api_key: The API key for authenticating with Cohere.
        model: The Cohere model to use. Note: 'rerank-multilingual-v3.0' 
               is the current state-of-the-art for cross-lingual tasks.
        top_n: The number of refined documents to return to the LLM.

    Returns:
        A ContextualCompressionRetriever instance integrated with Cohere Rerank.
    """
    
    # Initialize the reranker compressor
    compressor = CohereRerank(
        cohere_api_key=cohere_api_key, 
        model=model, 
        top_n=top_n
    )

    # Wrap the base retriever with the compressor
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

def instantiate_llm(
    provider: Literal["OpenAI", "Google", "HuggingFace"],
    api_key: str,
    model_name: str,
    temperature: float = 0.5,
    top_p: float = 0.95,
    streaming: bool = True
) -> Any:
    """
    Factory function to instantiate Language Models from various providers.

    Args:
        provider: The service provider ("OpenAI", "Google", or "HuggingFace").
        api_key: The authentication key for the chosen provider.
        model_name: The specific model ID (e.g., 'gpt-5.2', 'gemini-2.5-pro').
        temperature: Controls randomness (0.0 to 1.0). Default is 0.5.
        top_p: Nucleus sampling parameter. Default is 0.95.
        streaming: Whether to support real-time token streaming.

    Returns:
        An instance of a LangChain ChatModel or LLM.

    Raises:
        ValueError: If an unsupported provider is specified.
    """
    
# Core configuration shared across providers
    common_params = {
        "model": model_name,
        "temperature": temperature,
        "streaming": streaming,
        "top_p": top_p, # Now explicit
    }

    try:
        if provider == "OpenAI":
            return ChatOpenAI(
                api_key=api_key,
                **common_params
            )

        if provider == "Google":
            return ChatGoogleGenerativeAI(
                google_api_key=api_key,
                convert_system_message_to_human=True,
                **common_params
            )

        if provider == "HuggingFace":
            llm_endpoint = HuggingFaceEndpoint(
                repo_id=model_name,
                huggingfacehub_api_token=api_key,
                do_sample=True,
                max_new_tokens=1024,
                **common_params
            )
            return ChatHuggingFace(llm=llm_endpoint)

        raise ValueError(f"Unsupported LLM provider: {provider}")

    except Exception as e:
        # FAANG logging tip: Never just print; use structured logging in real apps
        print(f"[Error] Initialization failed for {provider}: {str(e)}")
        raise

def create_memory_config(
    model_name: str = "gpt-4o-mini", 
    max_token_limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Returns a modern configuration for conversation persistence.
    In 2026, we utilize LCEL Wrappers or LangGraph State instead of 
    legacy Memory classes.
    """
    
    # 1. Define the persistence layer (In-memory for this example)
    # For production, replace with RedisChatMessageHistory or MongoDBChatMessageHistory
    history_factory = {}

    def get_session_history(session_id: str):
        if session_id not in history_factory:
            history_factory[session_id] = InMemoryChatMessageHistory()
        return history_factory[session_id]

    # 2. Logic for Summarization (Modern approach)
    # Instead of a buffer class, we use a 'trimmer' or 'summarizer' 
    # as a component in our LCEL chain.
    
    memory_settings = {
        "get_session_history": get_session_history,
        "input_messages_key": "question",
        "history_messages_key": "chat_history",
        "summary_threshold": max_token_limit or 2048,
        "use_summarization": (model_name == "gpt-4o-mini")
    }

    return memory_settings