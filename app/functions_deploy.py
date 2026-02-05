from langchain_community.document_loaders import (
    TextLoader, 
    CSVLoader, 
    PyPDFLoader, 
    Docx2txtLoader
)


from langchain_core.embeddings import Embeddings


from typing import Optional, Literal, Dict, Any
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever

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

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.retrievers import BaseRetriever

from typing import Optional, Literal, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


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
        print(f"[Error] Initialization failed for {provider}: {str(e)}")
        raise