from dataclasses import dataclass, field
from typing import Dict, Any, List
import numpy as np
from dotenv import load_dotenv
load_dotenv()

@dataclass
class CaseDocument:
    """Case document with metadata"""
    id: str
    content: str
    vector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RetrievalResult:
    """Search result container"""
    case: CaseDocument
    score: float
    rank: int = 0