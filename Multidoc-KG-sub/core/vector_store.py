"""
Vector Store for Entity Semantic Search using Sentence Transformers and FAISS.
"""
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import faiss


class VectorStore:
    """Vector store for entity semantic search."""
    
    def __init__(self, model_name: str = 'BAAI/bge-m3'):
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
    
    def batch_add_entities(self, entities: List[Dict[str, str]]) -> None:
        """
        Add multiple entities to the vector store in one batch encode call.

        Args:
            entities: List of dicts with 'name' and 'id' keys
        """
        if not entities:
            return

        names = [e['name'] for e in entities]
        embeddings = self.model.encode(names, convert_to_numpy=True, batch_size=256).astype('float32')
        self.index.add(embeddings)
        self.metadata.extend(entities)
        print(f"[VectorStore] Batch added {len(entities)} entities, total: {len(self.metadata)}")
    
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
        actual_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, actual_k)
        
        # Build results
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                results.append({
                    'name': self.metadata[idx]['name'],
                    'id': self.metadata[idx]['id'],
                    'score': float(distance)
                })
        
        print(f"[VectorStore] Search for '{query}' returned {len(results)} results")
        return results
    
    def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[Dict[str, any]]]:
        """
        Batch search for multiple queries in one encode call.

        Args:
            queries: List of query strings
            top_k: Number of top results per query

        Returns:
            List of result lists, one per query
        """
        if self.index.ntotal == 0:
            return [[] for _ in queries]

        # Encode all queries in one batch call
        query_embeddings = self.model.encode(queries, convert_to_numpy=True, batch_size=256).astype('float32')

        actual_k = min(top_k, self.index.ntotal)
        all_distances, all_indices = self.index.search(query_embeddings, actual_k)

        all_results = []
        for distances, indices in zip(all_distances, all_indices):
            results = []
            for distance, idx in zip(distances, indices):
                if idx < len(self.metadata):
                    results.append({
                        'name': self.metadata[idx]['name'],
                        'id': self.metadata[idx]['id'],
                        'score': float(distance)
                    })
            all_results.append(results)

        return all_results
    
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
        self.model = None
        self.index = None
        self.metadata = []
        print("[MockVectorStore] Initialized (mock mode)")
    
    def add_entity(self, entity_name: str, entity_id: str) -> None:
        """Add entity to mock store."""
        self.entities[entity_id] = entity_name
        self.metadata.append({'name': entity_name, 'id': entity_id})
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
