# DocuMind AI

### Intelligent Document Insights Engine

![](https://github.com/gsdev42/Bajaj-Hack/blob/bec9e69f15270b038c6d8771fc73cfaddae78689/DocuMind%20AI.png)

> **"From pages to precision – transforming documents into decisions."**  
> An AI-powered system that extracts actionable insights from complex legal, insurance, and compliance documents.

---

## Why DocuMind AI?

- **Fast analysis** – Process insurance documents fast
- **Precision insights** – Domain-specific intelligence for insurance, legal, and compliance
- **Decision-ready outputs** – Structured JSON responses for immediate action
- **Context-aware intelligence** – Understands document nuances
- **Seamless integration** – API-ready for enterprise workflow integration

---

## Component Specifications

| Component          | Technology        | Key Features                                                   |
| ------------------ | ----------------- | -------------------------------------------------------------- |
| Query Processing   | GPT-3/BERT LLMs   | Semantic tokenization, Vector encoding (Algorithm 1)           |
| Dense Vector Index | BERT Architecture | 768-1024 dim vectors, HNSW indexing, Dynamic updates           |
| Retrieval Engine   | Qdrant + MMR      | Cosine similarity, Domain-aware ranking (Algorithm 2)          |
| Insight Generation | Gemini/Open AI    | Context aggregation, Conditional text generation (Algorithm 3) |

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Gemini AI](https://img.shields.io/badge/Gemini_AI-4285F4?logo=google&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-FF6D00?logo=qdrant&logoColor=white)  
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-FACE15)

---

## Quick Start

### Prerequisites

- Python 3.11+
- Qdrant database (Docker setup recommended)
- Gemini/OpenAI API key

### Installation

```bash
git clone https://github.com/yourusername/documind-ai.git

pip install -r requirements.txt

cp .env.example .env
nano .env

uvicorn app.main:app --reload --port 8000
```

### Sample API Request

```python
import requests

response = requests.post(
    "http://localhost:8000/analyze",
    json={
        "documents": ["https://example.com/policy.pdf"],
        "questions": [
            "What is the grace period for premium payments?",
            "Are pre-existing conditions covered?"
        ]
    },
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)

print(response.json())
```

---

## Architecture

```mermaid
graph TD
    A[User Query] --> B(Query Processing Module)
    B --> C[LLM Tokenization & Encoding]
    C --> D[Query Vector]
    D --> E{Case Retrieval Engine}

    F[Case Documents] --> G[BERT Vectorization]
    G --> H[Dense Vector Index]
    H --> E

    E --> I[HNSW Index Structure]
    I --> J[ Similarity Search]
    J --> K[MMR Diversification]
    K --> L[Top-k Relevant Cases]

    L --> M(Insight Generation Module)
    M --> N[Context Aggregation]
    N --> O[LLM Analysis]
    O --> P[JSON Insights]

    P --> Q[Decision Support Dashboard]
    Q --> R[Medical Diagnosis]
    Q --> S[Legal Precedents]

    subgraph Data Pipeline
        F --> G --> H
    end

    subgraph Core Engine
        B --> E --> M
    end

    subgraph Output
        Q
    end
```

---

## Sequence Diagram

```mermaid
sequenceDiagram
   User->>Query Processing: Submit natural language query
   Query Processing->>Retrieval Engine: Encoded vector
   Retrieval Engine->>Dense Vector Index: Similarity search
   Dense Vector Index-->>Retrieval Engine: Top 100 candidates
   Retrieval Engine->>Retrieval Engine: MMR diversification
   Retrieval Engine->>Insight Generation: Top-k cases + query
   Insight Generation->>LLM: Context aggregation
   LLM-->>Insight Generation: Analytical insights
   Insight Generation->>Dashboard: Structured JSON output
   Dashboard->>User: Decision support visualization
```

---

## Deployment

### Docker Setup

```Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
```

### Cloud Deployment

[![Deployed on Render](https://render.com/images/deploy-to-render-button.svg)](<[https://render.com/deploy](https://bajaj-hack-gydf.onrender.com)>)

---
