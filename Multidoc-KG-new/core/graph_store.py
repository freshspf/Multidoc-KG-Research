"""
Mock graph database store interface.
"""
from typing import List, Dict, Any, Optional


class MockGraphStore:
    """Mock graph database store for knowledge graph operations."""
    
    def __init__(self, connection_string: str = "mock://localhost:7687"):
        """
        Initialize mock graph store.
        
        Args:
            connection_string: Database connection string
        """
        self.connection_string = connection_string
        # Store entity mapping: entity_id -> entity_name for testing
        self.entity_mapping: Dict[str, str] = {}
        print(f"[MockGraphStore] Initialized with connection: {connection_string}")
    
    def query(self, cypher_query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query on the graph database.
        
        Args:
            cypher_query: Cypher query string
            **kwargs: Additional query parameters
            
        Returns:
            List of result records
        """
        # TODO: Implement actual graph query logic
        print(f"[MockGraphStore] query() called with query: {cypher_query[:100]}...")
        return []
    
    def vector_search(self, embedding: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search on graph nodes.
        
        Args:
            embedding: Vector embedding to search for
            top_k: Number of top results to return
            
        Returns:
            List of similar nodes with scores
        """
        # TODO: Implement vector search logic
        print(f"[MockGraphStore] vector_search() called with top_k={top_k}")
        return []
    
    def create_node(self, node_type: str, properties: Dict[str, Any]) -> str:
        """
        Create a new node in the graph.
        
        Args:
            node_type: Type/label of the node
            properties: Properties of the node
            
        Returns:
            ID of the created node
        """
        # TODO: Implement node creation logic
        print(f"[MockGraphStore] create_node() called: type={node_type}, properties={list(properties.keys())}")
        return f"mock_node_id_{node_type}"
    
    def create_relationship(self, from_id: str, to_id: str, rel_type: str, properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a relationship between two nodes.
        
        Args:
            from_id: Source node ID
            to_id: Target node ID
            rel_type: Type of relationship
            properties: Optional relationship properties
            
        Returns:
            ID of the created relationship
        """
        # TODO: Implement relationship creation logic
        print(f"[MockGraphStore] create_relationship() called: {from_id} -[{rel_type}]-> {to_id}")
        return f"mock_rel_id"
    
    def versioned_write(self, claim_id: str, version: int, data: Dict[str, Any]) -> bool:
        """
        Write a versioned claim to the graph with versioning support.
        
        Args:
            claim_id: Unique claim identifier
            version: Version number
            data: Claim data to write
            
        Returns:
            True if write was successful
        """
        # TODO: Implement versioned write logic
        print(f"[MockGraphStore] versioned_write() called: claim_id={claim_id}, version={version}")
        return True
    
    def get_entity_context(self, entity_id: str) -> List[str]:
        """
        Retrieve historical claims/context related to an entity.
        
        For testing purposes, this returns hardcoded conflicting knowledge
        for Chain-of-Thought related entities to test conflict detection.
        
        Args:
            entity_id: ID of the entity to get context for
            
        Returns:
            List of historical claim strings related to this entity
        """
        context = []
        
        # Hardcoded conflict for testing: Chain-of-Thought related entities
        # Check if the entity_id corresponds to CoT or related concepts
        cot_keywords = ["chain", "cot", "thought", "reasoning", "prompting"]
        
        # Simple check: if entity_id contains any CoT-related keywords
        entity_id_lower = entity_id.lower()
        if any(keyword in entity_id_lower for keyword in cot_keywords):
            # Return conflicting historical claim for testing
            context = [
                "Chain-of-Thought (CoT) was proven to degrade performance on arithmetic tasks in previous studies (2021).",
                "Studies showed that CoT prompting leads to inconsistent results and unreliable reasoning patterns."
            ]
        
        print(f"[MockGraphStore] Retrieved context for entity '{entity_id}': {len(context)} historical claims")
        if context:
            for idx, claim in enumerate(context, 1):
                print(f"[MockGraphStore]   Context {idx}: {claim[:80]}...")
        
        return context