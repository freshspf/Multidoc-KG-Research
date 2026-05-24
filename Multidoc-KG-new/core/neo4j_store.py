"""
Neo4j Graph Database Store implementation.
"""
import time
import re
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from schema import KnowledgeClaim


class Neo4jGraphStore:
    """Neo4j graph database store for knowledge graph operations."""
    
    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize Neo4j graph store with connection.
        
        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            user: Database username
            password: Database password
        """
        self.uri = uri
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print(f"[Neo4jGraphStore] Connected to Neo4j at {uri}")
        
        # Verify connection
        try:
            self.driver.verify_connectivity()
            print("[Neo4jGraphStore] Connection verified successfully")
        except Exception as e:
            print(f"[Neo4jGraphStore] WARNING: Connection verification failed: {e}")
    
    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            print("[Neo4jGraphStore] Connection closed")
    
    def _sanitize_relation_type(self, relation: str) -> str:
        """
        Sanitize relation string to be a valid Cypher relationship type.
        
        Args:
            relation: Raw relation string from claim
            
        Returns:
            Sanitized relation type (uppercase, spaces to underscores)
        """
        # Replace spaces with underscores, remove special chars, uppercase
        sanitized = re.sub(r'[^\w\s]', '', relation)  # Remove special chars
        sanitized = sanitized.replace(' ', '_')  # Replace spaces with underscores
        sanitized = sanitized.upper()  # Convert to uppercase
        return sanitized if sanitized else "RELATES_TO"
    
    def get_entity_context(self, entity_name: str) -> List[str]:
        """
        Retrieve historical claims/context related to an entity by name.
        
        Args:
            entity_name: Name of the entity to get context for
            
        Returns:
            List of historical claim strings related to this entity
        """
        context = []
        
        with self.driver.session() as session:
            # Query to find entity by name and get its relationships
            query = """
            MATCH (e:Entity)
            WHERE e.name = $name OR e.id = $name
            WITH e
            MATCH (e)-[r]-(neighbor:Entity)
            RETURN type(r) as rel_type, 
                   neighbor.name as neighbor_name,
                   r.evidence as evidence
            LIMIT 10
            """
            
            try:
                result = session.run(query, name=entity_name)
                
                for record in result:
                    rel_type = record["rel_type"]
                    neighbor_name = record["neighbor_name"]
                    evidence = record.get("evidence", "")
                    
                    # Format as natural language
                    context_str = f"Entity '{entity_name}' has relationship '{rel_type}' with '{neighbor_name}'"
                    if evidence:
                        context_str += f" (Evidence: {evidence[:100]}...)"
                    context.append(context_str)
                
                print(f"[Neo4jGraphStore] Retrieved {len(context)} historical claims for entity '{entity_name}'")
                if context:
                    for idx, claim in enumerate(context[:3], 1):  # Show first 3
                        print(f"[Neo4jGraphStore]   Context {idx}: {claim[:80]}...")
                
            except Exception as e:
                print(f"[Neo4jGraphStore] Error retrieving context for '{entity_name}': {e}")
        
        return context
    
    def write_claims(self, claims: List[KnowledgeClaim]) -> List[str]:
        """
        Write validated claims to the graph with versioning support.
        
        Args:
            claims: List of KnowledgeClaim objects to write
            
        Returns:
            List of claim IDs that were successfully written
        """
        written_ids = []
        
        with self.driver.session() as session:
            for claim in claims:
                try:
                    # Only write validated claims
                    if claim.status.value != "validated":
                        continue
                    
                    # Generate timestamp version
                    version = int(time.time() * 1000)  # Millisecond timestamp
                    
                    # Sanitize relation type for Cypher
                    rel_type = self._sanitize_relation_type(claim.relation)
                    
                    # Build Cypher query with dynamic relationship type
                    # Note: Cannot use parameters for relationship types in Cypher
                    query = f"""
                    // Create or merge subject node
                    MERGE (s:Entity {{id: $subject_id}})
                    ON CREATE SET s.name = $subject_name, s.created_at = timestamp()
                    ON MATCH SET s.updated_at = timestamp()
                    
                    // Create or merge object node
                    MERGE (o:Entity {{id: $object_id}})
                    ON CREATE SET o.name = $object_name, o.created_at = timestamp()
                    ON MATCH SET o.updated_at = timestamp()
                    
                    // Create relationship with versioning
                    CREATE (s)-[r:{rel_type}]->(o)
                    SET r.version = $version,
                        r.source_paper = $source_paper,
                        r.evidence = $evidence,
                        r.created_at = timestamp(),
                        r.relation_original = $relation_original
                    
                    RETURN s.id as subject_id, o.id as object_id, type(r) as rel_type
                    """
                    
                    # Execute query with parameters
                    result = session.run(
                        query,
                        subject_id=claim.subject_id or claim.subject,
                        subject_name=claim.subject,
                        object_id=claim.object_id or claim.object,
                        object_name=claim.object,
                        version=version,
                        source_paper=claim.source_paper_id,
                        evidence=claim.evidence,
                        relation_original=claim.relation
                    )
                    
                    # Get result
                    record = result.single()
                    if record:
                        claim_id = f"{claim.source_paper_id}_{record['subject_id']}_{record['object_id']}"
                        written_ids.append(claim_id)
                        print(f"[Neo4jGraphStore] ✓ Written: {claim.subject} -[{rel_type}]-> {claim.object[:40]}...")
                    
                except Exception as e:
                    print(f"[Neo4jGraphStore] ✗ Error writing claim: {e}")
                    print(f"[Neo4jGraphStore]   Claim: {claim.subject} {claim.relation} {claim.object}")
        
        print(f"[Neo4jGraphStore] Successfully wrote {len(written_ids)} claims to graph")
        return written_ids
    
    def query(self, cypher_query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query on the graph database.
        
        Args:
            cypher_query: Cypher query string
            **kwargs: Query parameters
            
        Returns:
            List of result records as dictionaries
        """
        results = []
        
        with self.driver.session() as session:
            try:
                result = session.run(cypher_query, **kwargs)
                results = [dict(record) for record in result]
                print(f"[Neo4jGraphStore] Query executed, returned {len(results)} results")
            except Exception as e:
                print(f"[Neo4jGraphStore] Query error: {e}")
        
        return results
    
    def clear_all(self):
        """
        Clear all nodes and relationships from the database.
        WARNING: This deletes all data!
        """
        with self.driver.session() as session:
            try:
                session.run("MATCH (n) DETACH DELETE n")
                print("[Neo4jGraphStore] All data cleared from database")
            except Exception as e:
                print(f"[Neo4jGraphStore] Error clearing database: {e}")
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with node and relationship counts
        """
        stats = {"nodes": 0, "relationships": 0}
        
        with self.driver.session() as session:
            try:
                # Count nodes
                result = session.run("MATCH (n) RETURN count(n) as count")
                stats["nodes"] = result.single()["count"]
                
                # Count relationships
                result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                stats["relationships"] = result.single()["count"]
                
                print(f"[Neo4jGraphStore] Stats: {stats['nodes']} nodes, {stats['relationships']} relationships")
            except Exception as e:
                print(f"[Neo4jGraphStore] Error getting stats: {e}")
        
        return stats

    # Graph Cleanup Methods
    # =========================================================================
    
    def run_garbage_collection(self) -> int:
        """
        Delete isolated nodes (Entity nodes with no relationships).
        
        Note: Ontology nodes are preserved even if they have no connections.
        
        Returns:
            Number of nodes deleted
        """
        with self.driver.session() as session:
            try:
                query = """
                MATCH (n:Entity)
                WHERE NOT (n)--()
                WITH n, n.name as name, n.id as id
                DELETE n
                RETURN count(*) as deleted_count
                """
                
                result = session.run(query)
                record = result.single()
                deleted_count = record["deleted_count"] if record else 0
                
                print(f"[Neo4jGraphStore] Garbage collection: removed {deleted_count} isolated nodes")
                return deleted_count
                
            except Exception as e:
                print(f"[Neo4jGraphStore] Error in garbage collection: {e}")
                return 0
    
    def merge_entities(self, primary_id: str, secondary_id: str) -> bool:
        """
        Merge two entities, transferring all relationships from secondary to primary.
        
        Args:
            primary_id: ID of the entity to keep
            secondary_id: ID of the entity to merge and delete
            
        Returns:
            True if successful, False otherwise
        """
        with self.driver.session() as session:
            try:
                check_query = """
                MATCH (p:Entity {id: $primary_id})
                MATCH (s:Entity {id: $secondary_id})
                RETURN p.id as primary, s.id as secondary
                """
                result = session.run(check_query, primary_id=primary_id, secondary_id=secondary_id)
                record = result.single()
                
                if not record:
                    print(f"[Neo4jGraphStore] One or both entities not found: {primary_id}, {secondary_id}")
                    return False
                
                # Manual relationship transfer (works without APOC)
                self._manual_merge_relationships(session, primary_id, secondary_id)
                
                # Delete secondary entity
                delete_query = """
                MATCH (secondary:Entity {id: $secondary_id})
                DETACH DELETE secondary
                """
                session.run(delete_query, secondary_id=secondary_id)
                
                print(f"[Neo4jGraphStore] Merged entity {secondary_id} into {primary_id}")
                return True
                
            except Exception as e:
                print(f"[Neo4jGraphStore] Error merging entities: {e}")
                return False
    
    def _manual_merge_relationships(self, session, primary_id: str, secondary_id: str):
        """Manually transfer relationships when APOC is not available."""
        get_rels_query = """
        MATCH (secondary:Entity {id: $secondary_id})-[r]-(other)
        WHERE NOT type(r) = 'IS_A' 
          AND other.id <> $primary_id
          AND other.id <> $secondary_id
        RETURN type(r) as rel_type, 
               startNode(r) = secondary as is_outgoing,
               other.id as other_id,
               properties(r) as props
        """
        
        result = session.run(get_rels_query, secondary_id=secondary_id, primary_id=primary_id)
        relationships = list(result)
        
        for record in relationships:
            rel_type = record["rel_type"]
            is_outgoing = record["is_outgoing"]
            other_id = record["other_id"]
            props = dict(record["props"]) if record["props"] else {}
            
            try:
                if is_outgoing:
                    create_query = f"""
                    MATCH (primary:Entity {{id: $primary_id}}), (other:Entity {{id: $other_id}})
                    WHERE NOT (primary)-[:{rel_type}]->(other)
                    CREATE (primary)-[r:{rel_type}]->(other)
                    SET r = $props
                    """
                else:
                    create_query = f"""
                    MATCH (primary:Entity {{id: $primary_id}}), (other:Entity {{id: $other_id}})
                    WHERE NOT (other)-[:{rel_type}]->(primary)
                    CREATE (other)-[r:{rel_type}]->(primary)
                    SET r = $props
                    """
                session.run(create_query, primary_id=primary_id, other_id=other_id, props=props)
            except Exception as e:
                print(f"[Neo4jGraphStore] Warning: Could not transfer relationship {rel_type}: {e}")
    
    def find_duplicate_entities(self, case_insensitive: bool = True) -> List[Dict[str, Any]]:
        """Find entities with duplicate names."""
        with self.driver.session() as session:
            try:
                if case_insensitive:
                    query = """
                    MATCH (e:Entity)
                    WITH toLower(e.name) as lower_name, collect({
                        id: e.id, name: e.name, entity_type: e.entity_type, created_at: e.created_at
                    }) as entities
                    WHERE size(entities) > 1
                    RETURN lower_name as group_name, entities, size(entities) as count
                    ORDER BY count DESC
                    """
                else:
                    query = """
                    MATCH (e:Entity)
                    WITH e.name as name, collect({
                        id: e.id, name: e.name, entity_type: e.entity_type, created_at: e.created_at
                    }) as entities
                    WHERE size(entities) > 1
                    RETURN name as group_name, entities, size(entities) as count
                    ORDER BY count DESC
                    """
                
                result = session.run(query)
                groups = []
                for record in result:
                    groups.append({
                        "group_name": record["group_name"],
                        "count": record["count"],
                        "entities": record["entities"]
                    })
                
                print(f"[Neo4jGraphStore] Found {len(groups)} groups of duplicate entities")
                return groups
                
            except Exception as e:
                print(f"[Neo4jGraphStore] Error finding duplicate entities: {e}")
                return []
