"""
Knowledge Evolution Agent: Handles versioned writing to the Graph DB.
"""
from typing import List, Union
from schema import KnowledgeClaim
from core.graph_store import MockGraphStore


class KnowledgeEvolutionAgent:
    """Agent responsible for writing validated claims to the graph with versioning."""
    
    def __init__(self, graph_store: Union[MockGraphStore, 'Neo4jGraphStore']):
        """
        Initialize Knowledge Evolution Agent.
        
        Args:
            graph_store: Graph database store for versioned writes (Mock or Neo4j)
        """
        self.graph_store = graph_store
        print("[KnowledgeEvolutionAgent] Initialized")
    
    def process(self, claims: List[KnowledgeClaim]) -> List[str]:
        """
        Write validated claims to the graph with versioning.
        
        Args:
            claims: List of validated KnowledgeClaim objects
            
        Returns:
            List of claim IDs that were successfully written
        """
        print(f"[KnowledgeEvolutionAgent] Processing {len(claims)} claims for graph evolution")
        
        # Check if graph_store has write_claims method (Neo4j) or use legacy methods (Mock)
        if hasattr(self.graph_store, 'write_claims'):
            # Use Neo4j batch write method
            written_ids = self.graph_store.write_claims(claims)
        else:
            # Use legacy Mock methods for backward compatibility
            written_ids = []
            for claim in claims:
                if claim.status.value == "validated":
                    # Create subject node
                    subject_id = self.graph_store.create_node(
                        node_type="Entity",
                        properties={"name": claim.subject}
                    )
                    
                    # Create object node
                    object_id = self.graph_store.create_node(
                        node_type="Entity",
                        properties={"name": claim.object}
                    )
                    
                    # Create relationship
                    rel_id = self.graph_store.create_relationship(
                        from_id=subject_id,
                        to_id=object_id,
                        rel_type=claim.relation,
                        properties={"evidence": claim.evidence, "source": claim.source_paper_id}
                    )
                    
                    # Versioned write
                    claim_id = f"{claim.source_paper_id}_{subject_id}_{object_id}"
                    self.graph_store.versioned_write(
                        claim_id=claim_id,
                        version=1,
                        data=claim.dict()
                    )
                    
                    written_ids.append(claim_id)
        
        print(f"[KnowledgeEvolutionAgent] Successfully wrote {len(written_ids)} claims to graph")
        return written_ids
