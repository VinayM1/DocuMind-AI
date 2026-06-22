from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from .models import CaseDocument
from .parser import parse_document
import numpy as np
from dotenv import load_dotenv
from typing import List
import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.http.exceptions import UnexpectedResponse

load_dotenv()

def index_to_qdrant(url: str, collection_name: str = "legal_cases") -> List[CaseDocument]:
    try:
        split_docs = parse_document(url)
        print(f"Created {len(split_docs)} chunks")

        embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )

        if not client.collection_exists(collection_name):
            print(f"Creating collection '{collection_name}'...")
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=3072,
                    distance=Distance.COSINE
                )
            )
        else:
            print(f"Collection '{collection_name}' already exists")

        try:
            print("Creating index for 'metadata.source'...")
            client.create_payload_index(
                collection_name=collection_name,
                field_name="metadata.source",
                field_schema="keyword"
            )
        except UnexpectedResponse as e:
            if "already exists" not in str(e):
                raise e
            print("Index for 'metadata.source' already exists")

        vector_store = QdrantVectorStore.from_documents(
            documents=split_docs,
            embedding=embedding_model,
            collection_name=collection_name,
            client=client
        )

        embeddings = embedding_model.embed_documents([doc.page_content for doc in split_docs])
        case_docs = []

        for i, (doc, emb) in enumerate(zip(split_docs, embeddings)):
            case_doc = CaseDocument(
                id=f"{url}-{i}",
                content=doc.page_content,
                vector=np.array(emb),
                metadata=doc.metadata
            )
            case_docs.append(case_doc)

        print("Indexing completed successfully!")
        return case_docs

    except Exception as e:
        print(f"Indexing error: {e}")
        return []