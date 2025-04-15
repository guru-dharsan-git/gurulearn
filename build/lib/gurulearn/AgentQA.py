from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os
import pandas as pd
from typing import List, Dict, Any, Optional, Union

class QAAgent:
    def __init__(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        page_content_fields: Union[str, List[str]],
        metadata_fields: Optional[List[str]] = None,
        llm_model: str = "llama3.2",
        k: int = 5,
        embedding_model: str = "mxbai-embed-large",
        db_location: str = "./faiss_langchain_db",
        collection_name: str = "documents",
        prompt_template: Optional[str] = None,
        system_prompt: str = "You are an expert in answering questions about the provided information.",
    ):
        """
        Initialize a QA agent with embeddings and retrieval capabilities using FAISS.
        
        Args:
            data: DataFrame or list of dictionaries containing the source data
            page_content_fields: Field(s) to use as document content
            metadata_fields: Fields to include as metadata
            llm_model: Ollama model to use for generation
            k: Number of documents to retrieve
            embedding_model: Ollama model to use for embeddings
            db_location: Directory to store vector database
            collection_name: Name of the collection in the vector store
            prompt_template: Custom prompt template (if None, a default will be used)
            system_prompt: System prompt describing the agent's role
        """
        self.llm_model = llm_model
        self.k = k
        self.db_location = db_location
        self.collection_name = collection_name
        self.system_prompt = system_prompt
        
        # Set up embeddings
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        
        # Check if FAISS index exists
        index_path = os.path.join(db_location, collection_name)
        if os.path.exists(index_path):
            # Load existing FAISS index
            self.vector_store = FAISS.load_local(
                folder_path=index_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True  # Required for local loading
            )
        else:
            # Create documents and initialize FAISS
            if data is not None:
                documents = self._prepare_documents(data, page_content_fields, metadata_fields)
                self.vector_store = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
                # Ensure the directory exists
                os.makedirs(db_location, exist_ok=True)
                # Save the index
                self.vector_store.save_local(os.path.join(db_location, collection_name))
            else:
                # Initialize empty FAISS index if no data provided
                self.vector_store = FAISS.from_texts(
                    texts=["placeholder"], 
                    embedding=self.embeddings
                )
        
        # Set up retriever
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": k}
        )
        
        # Set up LLM
        self.model = OllamaLLM(model=llm_model)
        
        # Set up prompt template
        if prompt_template is None:
            prompt_template = f"""
            {system_prompt}

            Here are some relevant documents: {{reviews}}

            Here is the question to answer: {{question}}
            """
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Create chain
        self.chain = self.prompt | self.model
    
    def _prepare_documents(self, data, page_content_fields, metadata_fields):
        """Create documents from the input data."""
        documents = []
        
        # Convert DataFrame to list of dicts if needed
        if isinstance(data, pd.DataFrame):
            data_list = data.to_dict(orient='records')
        else:
            data_list = data
        
        # Process content fields
        if isinstance(page_content_fields, str):
            page_content_fields = [page_content_fields]
        
        # Create documents
        for i, item in enumerate(data_list):
            # Combine content fields
            content = " ".join([str(item.get(field, "")) for field in page_content_fields])
            
            # Extract metadata
            metadata = {}
            if metadata_fields:
                for field in metadata_fields:
                    if field in item:
                        metadata[field] = item[field]
            
            # Add an ID to metadata for reference
            metadata['id'] = str(i)
            
            document = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(document)
        
        return documents
    
    def add_documents(self, data, page_content_fields, metadata_fields=None):
        """Add new documents to the existing vector store."""
        documents = self._prepare_documents(data, page_content_fields, metadata_fields)
        self.vector_store.add_documents(documents)
        # Save the updated index
        self.vector_store.save_local(os.path.join(self.db_location, self.collection_name))
        
    def query(self, question: str) -> str:
        """
        Query the agent with a question.
        
        Args:
            question: The question to ask
            
        Returns:
            The generated answer
        """
        reviews = self.retriever.invoke(question)
        result = self.chain.invoke({"reviews": reviews, "question": question})
        return result
    
    def interactive_mode(self):
        """Start an interactive query session."""
        while True:
            print("\n\n-------------------------------")
            question = input("Ask your question (q to quit): ")
            print("\n\n")
            if question.lower() == "q":
                break
            
            result = self.query(question)
            print(result)


# Example usage:
if __name__ == "__main__":
    # Load data
    df = pd.read_csv("data.csv")
    
    # Create agent
    restaurant_agent = QAAgent(
        data=df,
        page_content_fields=["Title", "Review"],
        metadata_fields=["Rating", "Date"],
        llm_model="llama3.2",
        k=5,
        embedding_model="mxbai-embed-large",
        db_location="./faiss_restaurant_db",
        collection_name="restaurant_reviews",
        system_prompt="You are an expert in answering questions about a pizza restaurant. be concise and clear and provide details in points "
    )
    
    # Start interactive mode
    restaurant_agent.interactive_mode()
    ans = restaurant_agent.query("whats the best rated pizza shop?")
    print(ans)