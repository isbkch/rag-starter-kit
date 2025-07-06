# 5. Vector Database Abstraction Layer

This document explains the design and implementation of the Vector Database Abstraction Layer. This layer is a critical component of the platform, enabling flexibility and interchangeability of different vector database providers.

## Purpose and Design Goals

The primary goal of the abstraction layer is to decouple the application's core logic from the specific implementation details of any single vector database. This achieves several key objectives:

* **Flexibility**: Allows the platform to support multiple vector database backends (e.g., ChromaDB, Pinecone, Weaviate) without changing the application code.
* **Interchangeability**: System administrators can switch between providers by simply changing an environment variable (`VECTOR_DB_PROVIDER`).
* **Maintainability**: Centralizes the logic for interacting with vector databases, making it easier to update or fix issues related to a specific provider.
* **Extensibility**: Provides a clear pattern for adding support for new vector database providers in the future.

## Implementation

The abstraction layer is implemented using two key design patterns: a **Base Interface** and a **Factory Pattern**.

### 1. Base Interface (`BaseVectorDB`)

A common interface is defined in `app/services/vectordb/base.py`. The `BaseVectorDB` class is an abstract base class (ABC) that declares a set of common methods that any concrete vector database implementation must provide. These methods include:

* `connect()`: Establishes a connection to the database.
* `disconnect()`: Closes the connection.
* `create_collection()`: Creates a new collection or index for storing vectors.
* `upsert_documents()`: Adds or updates documents (chunks and their vectors) in the database.
* `query()`: Performs a similarity search for a given query vector.
* `delete_documents()`: Deletes documents from the database.

### 2. Concrete Implementations

For each supported vector database, a concrete class is created in the `app/services/vectordb/` directory. This class inherits from `BaseVectorDB` and implements all the abstract methods using the specific client library and API of that provider.

* **ChromaDB**: `app/services/vectordb/chroma_db.py`
* **Pinecone**: `app/services/vectordb/pinecone_db.py`
* **Weaviate**: `app/services/vectordb/weaviate_db.py`

### 3. Factory Pattern (`VectorDBFactory`)

A factory function, `get_vector_db`, is defined in `app/services/vectordb/factory.py`. This function is responsible for instantiating and returning the correct vector database client based on the `VECTOR_DB_PROVIDER` environment variable.

When a service needs to interact with the vector database, it calls the `get_vector_db()` factory. The factory reads the configuration, determines which provider is selected, and returns an instance of the corresponding concrete class (e.g., `ChromaDBClient`).

Because all concrete clients adhere to the `BaseVectorDB` interface, the calling service can use the returned instance without needing to know which specific database is being used.

## Usage Example

```python
# In document_service.py or another service

from app.services.vectordb.factory import get_vector_db

def process_and_store_document(doc):
    # The factory returns the configured DB client (Chroma, Pinecone, etc.)
    vector_db_client = get_vector_db()

    # The service code remains the same regardless of the provider
    vector_db_client.connect()
    vector_db_client.upsert_documents(documents=[...])
    vector_db_client.disconnect()
```

This design ensures that the rest of the application is completely agnostic to the underlying vector database, making the system highly modular and adaptable.
