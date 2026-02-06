"""
AgentQA - RAG-based Question Answering with Vector Stores.

Provides a QA agent that uses embeddings and retrieval-augmented generation
to answer questions based on provided data.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


class QueryResult(TypedDict):
    """Result from a query operation."""
    answer: str
    sources: list[dict[str, Any]]
    query: str


class QAAgent:
    """
    RAG-based Question Answering Agent using FAISS and Ollama.
    
    Args:
        data: DataFrame or list of dictionaries containing source data
        page_content_fields: Field(s) to use as document content
        metadata_fields: Fields to include as metadata (optional)
        llm_model: Ollama model for generation (default: "llama3.2")
        k: Number of documents to retrieve (default: 5)
        embedding_model: Ollama model for embeddings (default: "mxbai-embed-large")
        db_location: Directory for vector database storage
        collection_name: Name of the collection
        prompt_template: Custom prompt template (optional)
        system_prompt: System prompt for the agent
        
    Example:
        >>> import pandas as pd
        >>> df = pd.read_csv("reviews.csv")
        >>> agent = QAAgent(
        ...     data=df,
        ...     page_content_fields=["title", "review"],
        ...     metadata_fields=["rating", "date"],
        ...     system_prompt="You are a helpful assistant."
        ... )
        >>> result = agent.query("What are the best rated items?")
        >>> print(result["answer"])
    """

    DEFAULT_TEMPLATE = """
{system_prompt}

Here are some relevant documents:
{reviews}

Based on the above information, please answer the following question:
{question}

Provide a clear and concise answer.
"""

    def __init__(
        self,
        data: pd.DataFrame | list[dict[str, Any]] | None = None,
        page_content_fields: str | list[str] = "",
        metadata_fields: list[str] | None = None,
        llm_model: str = "llama3.2",
        k: int = 5,
        embedding_model: str = "mxbai-embed-large",
        db_location: str | Path = "./faiss_langchain_db",
        collection_name: str = "documents",
        prompt_template: str | None = None,
        system_prompt: str = "You are an expert in answering questions about the provided information.",
    ):
        self.llm_model = llm_model
        self.k = k
        self.db_location = Path(db_location)
        self.collection_name = collection_name
        self.system_prompt = system_prompt
        
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        
        # Load or create vector store
        index_path = self.db_location / collection_name
        if index_path.exists():
            self.vector_store = FAISS.load_local(
                folder_path=str(index_path),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
        elif data is not None:
            documents = self._prepare_documents(data, page_content_fields, metadata_fields)
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            self.db_location.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(str(index_path))
        else:
            raise ValueError("Either data must be provided or existing index must exist at db_location")
        
        # Set up retriever
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": k}
        )
        
        # Set up LLM
        self.model = OllamaLLM(model=llm_model)
        
        # Set up prompt
        template = prompt_template or self.DEFAULT_TEMPLATE
        self.prompt = ChatPromptTemplate.from_template(template)
        
        # Create chain
        self.chain = self.prompt | self.model

    def _prepare_documents(
        self,
        data: pd.DataFrame | list[dict[str, Any]],
        page_content_fields: str | list[str],
        metadata_fields: list[str] | None
    ) -> list[Document]:
        """Create Document objects from input data."""
        documents = []
        
        # Convert DataFrame to list of dicts
        if isinstance(data, pd.DataFrame):
            data_list = data.to_dict(orient="records")
        else:
            data_list = data
        
        # Normalize content fields
        if isinstance(page_content_fields, str):
            page_content_fields = [page_content_fields]
        
        for i, item in enumerate(data_list):
            # Combine content fields
            content_parts = []
            for field in page_content_fields:
                value = item.get(field, "")
                if value and pd.notna(value):
                    content_parts.append(str(value))
            
            content = " ".join(content_parts)
            
            if not content.strip():
                continue
            
            # Extract metadata
            metadata = {"id": str(i)}
            if metadata_fields:
                for field in metadata_fields:
                    if field in item:
                        value = item[field]
                        # Convert to JSON-serializable type
                        if pd.notna(value):
                            metadata[field] = str(value) if not isinstance(value, (int, float, bool)) else value
            
            documents.append(Document(page_content=content, metadata=metadata))
        
        return documents

    def add_documents(
        self,
        data: pd.DataFrame | list[dict[str, Any]],
        page_content_fields: str | list[str],
        metadata_fields: list[str] | None = None
    ) -> int:
        """
        Add new documents to the vector store.
        
        Args:
            data: New data to add
            page_content_fields: Field(s) to use as content
            metadata_fields: Fields to include as metadata
            
        Returns:
            Number of documents added
        """
        documents = self._prepare_documents(data, page_content_fields, metadata_fields)
        self.vector_store.add_documents(documents)
        
        # Save updated index
        index_path = self.db_location / self.collection_name
        self.vector_store.save_local(str(index_path))
        
        return len(documents)

    def similarity_search(self, query: str, k: int | None = None) -> list[Document]:
        """
        Perform direct similarity search without LLM generation.
        
        Args:
            query: Search query
            k: Number of results (uses default if None)
            
        Returns:
            List of similar documents
        """
        return self.vector_store.similarity_search(query, k=k or self.k)

    def query(self, question: str, return_sources: bool = False) -> QueryResult | str:
        """
        Query the agent with a question.
        
        Args:
            question: The question to ask
            return_sources: Whether to return source documents
            
        Returns:
            QueryResult dict if return_sources=True, otherwise just the answer string
        """
        retrieved_docs = self.retriever.invoke(question)
        
        # Format documents for the prompt
        reviews_text = "\n\n".join([
            f"Document {i+1}:\n{doc.page_content}"
            for i, doc in enumerate(retrieved_docs)
        ])
        
        # Generate answer
        result = self.chain.invoke({
            "system_prompt": self.system_prompt,
            "reviews": reviews_text,
            "question": question
        })
        
        if return_sources:
            return QueryResult(
                answer=result,
                sources=[{"content": doc.page_content, "metadata": doc.metadata} for doc in retrieved_docs],
                query=question
            )
        
        return result

    def interactive_mode(self) -> None:
        """Start an interactive query session."""
        print("\n=== Interactive QA Mode ===")
        print("Type 'q' or 'quit' to exit\n")
        
        while True:
            try:
                question = input("Ask: ").strip()
                if question.lower() in ("q", "quit", "exit"):
                    print("Goodbye!")
                    break
                
                if not question:
                    continue
                
                result = self.query(question)
                print(f"\nAnswer: {result}\n")
                print("-" * 40 + "\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break

    def clear_index(self) -> None:
        """Delete the vector store index."""
        import shutil
        index_path = self.db_location / self.collection_name
        if index_path.exists():
            shutil.rmtree(index_path)
            print(f"Deleted index at {index_path}")