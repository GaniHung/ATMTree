from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

@dataclass(eq=False)
class ATMNode:
    retrieval_embedding: np.ndarray
    content_embedding: np.ndarray
    id: Optional[int] = None
    is_leaf: bool = False
    children: List['ATMNode'] = field(default_factory=list)
    parent: Optional['ATMNode'] = None
    data_points: List[int] = field(default_factory=list)
    num_embeddings: int = 1
    generation_method: str = ""

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, ATMNode):
            return False
        return self.id == other.id

@dataclass
class ATMTree:
    root: ATMNode
