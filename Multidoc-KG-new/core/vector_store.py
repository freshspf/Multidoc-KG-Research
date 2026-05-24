"""
Vector Store for Entity Semantic Search using Sentence Transformers and FAISS.
"""
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import faiss


class VectorStore:
    """Vector store for entity semantic search."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize Vector Store with Sentence Transformer and FAISS index.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        print(f"[VectorStore] Initializing with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index (L2 distance)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Metadata storage: maps FAISS index position to entity info
        self.metadata: List[Dict[str, str]] = []
        
        print(f"[VectorStore] Initialized with embedding dimension: {self.embedding_dim}")
    
    def add_entity(self, entity_name: str, entity_id: str) -> None:
        """
        Add an entity to the vector store.
        
        Args:
            entity_name: Name of the entity (used for embedding)
            entity_id: Unique identifier for the entity
        """
        # Encode entity name to vector
        embedding = self.model.encode([entity_name], convert_to_numpy=True)
        
        # Add to FAISS index
        self.index.add(embedding.astype('float32'))
        
        # Store metadata
        self.metadata.append({
            'name': entity_name,
            'id': entity_id
        })
        
        print(f"[VectorStore] Added entity: '{entity_name}' (ID: {entity_id}), Total entities: {len(self.metadata)}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, any]]:
        """
        Search for similar entities in the vector store.
        
        Args:
            query: Query string to search for
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with keys: 'name', 'id', 'score' (lower is better for L2)
        """
        # Check if index is empty
        if self.index.ntotal == 0:
            print(f"[VectorStore] Index is empty, no results for query: '{query}'")
            return []
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype('float32')
        
        # Search in FAISS index
        # Limit top_k to available entities
        actual_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, actual_k)
        
        # Build results
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):  # Valid index
                results.append({
                    'name': self.metadata[idx]['name'],
                    'id': self.metadata[idx]['id'],
                    'score': float(distance)  # L2 distance (lower is better)
                })
        
        print(f"[VectorStore] Search for '{query}' returned {len(results)} results")
        return results
    
    def size(self) -> int:
        """Return the number of entities in the store."""
        return len(self.metadata)
    
    def clear(self) -> None:
        """Clear all entities from the store."""
        self.index.reset()
        self.metadata.clear()
        print("[VectorStore] Cleared all entities")


class MockVectorStore:
    """Mock vector store for testing without dependencies."""
    
    def __init__(self):
        """Initialize mock vector store."""
        self.entities = {}
        print("[MockVectorStore] Initialized (mock mode)")
    
    def add_entity(self, entity_name: str, entity_id: str) -> None:
        """Add entity to mock store."""
        self.entities[entity_id] = entity_name
        print(f"[MockVectorStore] Added entity: '{entity_name}' (ID: {entity_id})")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, any]]:
        """Mock search - returns empty list."""
        print(f"[MockVectorStore] Mock search for: '{query}'")
        return []
    
    def size(self) -> int:
        """Return mock size."""
        return len(self.entities)
    
    def clear(self) -> None:
        """Clear mock store."""
        self.entities.clear()
        print("[MockVectorStore] Cleared")
