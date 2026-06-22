from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from .models import CaseDocument, RetrievalResult
import numpy as np
from typing import List, Optional
from dotenv import load_dotenv
import os
load_dotenv()

class QdrantRetrievalEngine:
    def __init__(self, 
                 embedding_model,
                 collection_name: str = "legal_cases"):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        self.embedding_model = embedding_model
        self.collection_name = collection_name
    
    def retrieve_cases(self,
                      query: str,
                      k: int = 5,
                      diversity: float = 0.7,
                      source_filter: Optional[str] = None) -> List[RetrievalResult]:
        query_vector = self.embedding_model.embed_query(query)
        
        qdrant_filter = None
        if source_filter:
            qdrant_filter = Filter(
                must=[FieldCondition(
                    key="metadata.source", 
                    match=MatchValue(value=source_filter)
                )]
            )
        
        candidates = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=100,
            with_payload=True,
            with_vectors=True
        )
        candidate_results = []
        for hit in candidates:
            case_doc = CaseDocument(
                id=str(hit.id),
                content=hit.payload.get("page_content",""),
                vector=np.array(hit.vector),
                metadata=hit.payload.get("metadata", {})
            )
            candidate_results.append(
                RetrievalResult(case=case_doc, score=hit.score)
            )
        
        return self._apply_mmr(candidate_results, query_vector, k, diversity)
    
    def _apply_mmr(self, 
                  candidates: List[RetrievalResult], 
                  query_vector: np.ndarray,
                  k: int,
                  diversity: float) -> List[RetrievalResult]:
        results = []
        selected_vectors = []
        
        query_norm = np.linalg.norm(query_vector)
        if query_norm > 0:
            query_vector = query_vector / query_norm
        
        for rank in range(1, k + 1):
            if not candidates:
                break

            best_score = -10_000
            best_idx = -1

            for idx, candidate in enumerate(candidates):
                candidate_vector = candidate.case.vector
                candidate_norm = np.linalg.norm(candidate_vector)
                if candidate_norm > 0:
                    candidate_vector = candidate_vector / candidate_norm
                
                rel_score = np.dot(candidate_vector, query_vector)

                div_penalty = 0
                if selected_vectors:
                    similarities = np.dot(selected_vectors, candidate_vector)
                    div_penalty = np.max(similarities)

                mmr_score = diversity * rel_score - (1 - diversity) * div_penalty

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx >= 0:
                best = candidates.pop(best_idx)
                best.rank = rank
                best.score = best_score
                results.append(best)
                
                selected_vectors.append(best.case.vector / np.linalg.norm(best.case.vector))

        return results