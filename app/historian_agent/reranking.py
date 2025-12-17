#!/usr/bin/env python3
import logging
import os
from typing import List, Any
import numpy as np
from sentence_transformers import CrossEncoder

class DocumentReranker:
    def __init__(self, cross_encoder_weight=0.85, temporal_weight=0.10, entity_weight=0.05):
        model_name = os.environ.get("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.model = CrossEncoder(model_name)
        self.ce_w = cross_encoder_weight
        self.tp_w = temporal_weight
        self.et_w = entity_weight

    def rerank(self, query: str, documents: List[Any], top_k: int = 10):
        if not documents: return []
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)
        
        # Normalize scores to [0,1]
        scores = 1.0 / (1.0 + np.exp(-scores))
        
        scored_docs = []
        for i, doc in enumerate(documents):
            # For simplicity, using CE score primarily
            final_score = (self.ce_w * scores[i]) + (self.tp_w * 0.5) + (self.et_w * 0.5)
            doc.metadata["rerank_score"] = float(final_score)
            scored_docs.append(doc)
            
        scored_docs.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)
        return scored_docs[:top_k]