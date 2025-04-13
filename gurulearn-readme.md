# GuruLearn

## QAAgent: Create intelligent QA systems with RAG

GuruLearn's QAAgent provides a simple yet powerful way to create domain-specific question-answering agents using Retrieval Augmented Generation (RAG).

## Installation

```bash
pip install gurulearn
```

## Dependencies

GuruLearn requires the following dependencies:

- langchain-ollama
- langchain-core
- langchain-chroma
- pandas
- chromadb

Install them using:

```bash
pip install langchain-ollama langchain-core langchain-chroma pandas chromadb
```

## Quick Start

```python
from gurulearn import QAAgent
import pandas as pd

# Load your data
df = pd.read_csv("customer_support_tickets.csv")

# Create an agent
support_agent = QAAgent(
    data=df,
    page_content_fields=["Title", "Description"],
    metadata_fields=["Category", "Priority"],
    system_prompt="You are a helpful customer support agent."
)

# Query the agent
answer = support_agent.query("How do I reset my password?")
print(answer)

# Or use interactive mode
support_agent.interactive_mode()
```

## Features

- **Simple Setup**: Create powerful RAG-based QA systems with minimal code
- **Flexible Data Support**: Works with pandas DataFrames or lists of dictionaries
- **Custom Prompting**: Define system prompts and instructions to shape agent responses
- **Vector Database Integration**: Automatically creates and manages embeddings for efficient retrieval
- **Interactive Mode**: Built-in console interface for quick experimentation

## API Reference

### QAAgent

```python
QAAgent(
    data,                    # DataFrame or list of dictionaries containing the source data
    page_content_fields,     # Field(s) to use as document content
    metadata_fields=None,    # Fields to include as metadata
    llm_model="llama3.2",    # Ollama model to use for generation
    k=5,                     # Number of documents to retrieve
    embedding_model="mxbai-embed-large",  # Ollama model for embeddings
    db_location="./chroma_langchain_db",  # Directory to store vector database
    collection_name="documents",          # Name of the collection in the vector store
    prompt_template=None,    # Custom prompt template (if None, a default will be used)
    system_prompt="You are an expert in answering questions about the provided information."
)
```

#### Methods

- **query(question)**: Query the agent with a question and get a response
- **interactive_mode()**: Start an interactive console session for querying the agent

## Examples

### Restaurant Review QA System

```python
from gurulearn import QAAgent
import pandas as pd

# Load restaurant review data
df = pd.read_csv("restaurant_reviews.csv")

# Create a restaurant review agent
restaurant_agent = QAAgent(
    data=df,
    page_content_fields=["Title", "Review"],
    metadata_fields=["Rating", "Date"],
    llm_model="llama3.2",
    k=5,
    db_location="./chroma_restaurant_db",
    collection_name="restaurant_reviews",
    system_prompt="You are an expert in answering questions about a pizza restaurant."
)

# Ask questions about the restaurant
result = restaurant_agent.query("What do customers say about the pepperoni pizza?")
print(result)
```

### HR Policy Assistant

```python
from gurulearn import QAAgent

# Create an HR policy assistant
hr_documents = [
    {"Policy": "Parental Leave", "Description": "Employees are entitled to 12 weeks of paid parental leave...", "Department": "HR", "LastUpdated": "2023-09-01"},
    {"Policy": "Remote Work", "Description": "Employees may work remotely up to 3 days per week...", "Department": "HR", "LastUpdated": "2023-10-15"},
    # More policy documents...
]

hr_agent = QAAgent(
    data=hr_documents,
    page_content_fields=["Policy", "Description"],
    metadata_fields=["Department", "LastUpdated"],
    db_location="./chroma_hr_db",
    collection_name="hr_policies",
    system_prompt="You are an HR assistant providing information about company policies."
)

# Query the HR assistant
hr_agent.interactive_mode()
```

## Advanced Usage

### Custom Prompt Templates

You can define custom prompt templates to control how the agent processes and responds to queries:

```python
custom_template = """
You are a technical support specialist for computer hardware.

CONTEXT:
{reviews}

USER QUESTION:
{question}

Please provide a concise answer focusing only on the information found in the context.
If the information isn't in the context, admit you don't know.
"""

support_agent = QAAgent(
    data=support_df,
    page_content_fields=["Issue", "Resolution"],
    prompt_template=custom_template
)
```

### Using with Different Models

QAAgent works with any Ollama model:

```python
# Using with different Ollama models
medical_agent = QAAgent(
    data=medical_data,
    page_content_fields="text",
    llm_model="nous-hermes2:Q5_K_M", 
    embedding_model="nomic-embed-text"
)
```

## License

MIT
