"""
Neo4j Graph Database Store implementation with ontology support.
"""
import time
import re
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from schema import KnowledgeClaim, Paper, SubdomainAssignment


class Neo4jGraphStore:
    """Neo4j graph database store for knowledge graph operations with ontology support."""
    
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
            
            # Create ontology indexes
            self.create_ontology_indexes()
            
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
        允许中文，仅将空格替换为下划线，去除可能干扰的标点符号。
        
        Args:
            relation: Raw relation string from claim
            
        Returns:
            Sanitized relation type (可包含中文，空格转为下划线)
        """
        if not relation:
            return "关系"
        # 将空格替换为下划线
        sanitized = relation.strip().replace(' ', '_')
        # 去除可能引起问题的字符，保留汉字、字母、数字、下划线
        sanitized = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9_]', '', sanitized)
        return sanitized if sanitized else "关系"
    
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
            MATCH (e)
            WHERE e.name = $name OR e.id = $name
            WITH e
            OPTIONAL MATCH (e)-[r]-(neighbor)
            RETURN labels(e)[0] as node_type,
                   type(r) as rel_type, 
                   neighbor.name as neighbor_name,
                   r.evidence as evidence
            LIMIT 10
            """
            
            try:
                result = session.run(query, name=entity_name)
                
                for record in result:
                    node_type = record["node_type"]
                    rel_type = record["rel_type"]
                    neighbor_name = record["neighbor_name"]
                    evidence = record.get("evidence", "")
                    
                    if rel_type and neighbor_name:
                        # Format as natural language
                        context_str = f"[{node_type}] '{entity_name}' --{rel_type}--> '{neighbor_name}'"
                        if evidence:
                            context_str += f" (Evidence: {evidence[:100]}...)"
                        context.append(context_str)
                
                print(f"[Neo4jGraphStore] Retrieved {len(context)} historical claims for entity '{entity_name}'")
                
            except Exception as e:
                print(f"[Neo4jGraphStore] Error retrieving context for '{entity_name}': {e}")
        
        return context

    def get_entity_contexts_batch(self, entity_names: List[str]) -> Dict[str, List[str]]:
        """Retrieve historical contexts for multiple entity names in one query."""
        unique_names = [name for name in dict.fromkeys(entity_names) if name]
        if not unique_names:
            return {}

        contexts: Dict[str, List[str]] = {name: [] for name in unique_names}

        with self.driver.session() as session:
            try:
                result = session.run(
                    """
                    UNWIND $names AS target_name
                    MATCH (e)
                    WHERE e.name = target_name OR e.id = target_name
                    OPTIONAL MATCH (e)-[r]-(neighbor)
                    RETURN target_name,
                           labels(e)[0] AS node_type,
                           type(r) AS rel_type,
                           neighbor.name AS neighbor_name,
                           r.evidence AS evidence
                    LIMIT 200
                    """,
                    names=unique_names,
                )

                for record in result:
                    target_name = record["target_name"]
                    rel_type = record["rel_type"]
                    neighbor_name = record["neighbor_name"]
                    if not rel_type or not neighbor_name:
                        continue

                    node_type = record["node_type"]
                    evidence = record.get("evidence", "")
                    context_str = f"[{node_type}] '{target_name}' --{rel_type}--> '{neighbor_name}'"
                    if evidence:
                        context_str += f" (Evidence: {evidence[:100]}...)"
                    contexts.setdefault(target_name, []).append(context_str)
            except Exception as e:
                print(f"[Neo4jGraphStore] Error retrieving batch context: {e}")

        return contexts
    
    def create_ontology_indexes(self):
        """Create specialized indexes for ontology queries."""
        with self.driver.session() as session:
            try:
                # Index for Concept nodes
                session.run("CREATE INDEX concept_name IF NOT EXISTS FOR (c:Concept) ON (c.name)")
                session.run("CREATE INDEX concept_id IF NOT EXISTS FOR (c:Concept) ON (c.id)")
                
                # Index for Entity nodes
                session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")
                session.run("CREATE INDEX entity_id IF NOT EXISTS FOR (e:Entity) ON (e.id)")
                session.run("CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)")
                
                # Index for ontology properties
                session.run("CREATE INDEX is_concept IF NOT EXISTS FOR (n:Entity) ON (n.is_concept)")
                
                # Indexes for paper/subdomain organization
                session.run("CREATE INDEX paper_id IF NOT EXISTS FOR (p:Paper) ON (p.id)")
                session.run("CREATE INDEX paper_subdomain IF NOT EXISTS FOR (p:Paper) ON (p.subdomain)")
                session.run("CREATE INDEX subdomain_name IF NOT EXISTS FOR (s:Subdomain) ON (s.name)")
                session.run("CREATE INDEX subdomain_normalized_name IF NOT EXISTS FOR (s:Subdomain) ON (s.normalized_name)")
                session.run("CREATE INDEX subdomain_status IF NOT EXISTS FOR (s:Subdomain) ON (s.status)")
                session.run("CREATE INDEX candidate_name IF NOT EXISTS FOR (c:SubdomainCandidate) ON (c.name)")
                session.run("CREATE INDEX candidate_normalized_name IF NOT EXISTS FOR (c:SubdomainCandidate) ON (c.normalized_name)")
                session.run("CREATE INDEX candidate_batch_id IF NOT EXISTS FOR (c:SubdomainCandidate) ON (c.source_batch_id)")
                
                print("[Neo4jGraphStore] Ontology indexes created successfully")
            except Exception as e:
                print(f"[Neo4jGraphStore] Error creating indexes: {e}")

    def _normalize_subdomain_name(self, name: str) -> str:
        """Normalize subdomain labels for stable matching."""
        normalized = re.sub(r"\s+", " ", (name or "").strip().lower())
        return normalized.strip(" .;:,")

    def ensure_subdomain_root(self, root_name: str) -> None:
        """Ensure the top-level root subdomain node exists."""
        normalized = self._normalize_subdomain_name(root_name)
        if not normalized:
            return

        with self.driver.session() as session:
            try:
                session.run(
                    """
                    MERGE (root:Subdomain {normalized_name: $normalized_name})
                    ON CREATE SET root.created_at = timestamp()
                    SET root.updated_at = timestamp(),
                        root.name = $display_name,
                        root.is_root = true,
                        root.status = 'confirmed',
                        root.taxonomy_version = 1
                    """,
                    normalized_name=normalized,
                    display_name=root_name,
                )
            except Exception as e:
                print(f"[Neo4jGraphStore] Error ensuring subdomain root {root_name}: {e}")

    def write_paper_node(self, paper: Paper) -> None:
        """Upsert a Paper node with the latest subdomain metadata."""
        metadata = paper.metadata or {}
        abstract = paper.get_abstract()
        keywords = paper.get_keywords()
        subdomain = str(metadata.get("subdomain", "")).strip()
        parent_domain = str(metadata.get("parent_domain", "")).strip()
        reason = str(metadata.get("subdomain_reason", "")).strip()
        confidence = metadata.get("subdomain_confidence")
        subdomain_status = str(metadata.get("subdomain_status", "")).strip()
        batch_id = str(metadata.get("subdomain_batch_id", "")).strip()
        taxonomy_version = metadata.get("taxonomy_version")

        with self.driver.session() as session:
            try:
                session.run(
                    """
                    MERGE (p:Paper {id: $paper_id})
                    ON CREATE SET p.created_at = timestamp()
                    SET p.updated_at = timestamp(),
                        p.title = $title,
                        p.abstract = $abstract,
                        p.keywords = $keywords,
                        p.subdomain = $subdomain,
                        p.parent_domain = $parent_domain,
                        p.subdomain_reason = $subdomain_reason,
                        p.subdomain_confidence = $subdomain_confidence,
                        p.subdomain_status = $subdomain_status,
                        p.batch_id = $batch_id,
                        p.taxonomy_version = $taxonomy_version
                    """,
                    paper_id=paper.id,
                    title=paper.title,
                    abstract=abstract,
                    keywords=keywords,
                    subdomain=subdomain,
                    parent_domain=parent_domain,
                    subdomain_reason=reason,
                    subdomain_confidence=confidence,
                    subdomain_status=subdomain_status,
                    batch_id=batch_id,
                    taxonomy_version=taxonomy_version,
                )
            except Exception as e:
                print(f"[Neo4jGraphStore] Error writing Paper node {paper.id}: {e}")

    def write_subdomain_hierarchy(self, relations: List[Dict[str, str]], source: str = "llm") -> int:
        """Upsert Subdomain nodes and their SUBCLASS_OF relationships."""
        written = 0

        with self.driver.session() as session:
            for relation in relations:
                subject = self._normalize_subdomain_name(str(relation.get("subject", "")))
                rel = self._normalize_subdomain_name(str(relation.get("relation", "")))
                obj = self._normalize_subdomain_name(str(relation.get("object", "")))

                if not subject or not obj or rel != "subclass_of" or subject == obj:
                    continue

                try:
                    result = session.run(
                        """
                        MERGE (child:Subdomain {normalized_name: $child_norm})
                        ON CREATE SET child.created_at = timestamp()
                        SET child.updated_at = timestamp(),
                            child.name = $child_name,
                            child.status = 'confirmed'

                        MERGE (parent:Subdomain {normalized_name: $parent_norm})
                        ON CREATE SET parent.created_at = timestamp()
                        SET parent.updated_at = timestamp(),
                            parent.name = $parent_name,
                            parent.status = 'confirmed'

                        MERGE (child)-[r:SUBCLASS_OF]->(parent)
                        ON CREATE SET r.created_at = timestamp()
                        SET r.updated_at = timestamp(),
                            r.source = $source

                        RETURN child.normalized_name AS child_name
                        """,
                        child_norm=subject,
                        child_name=subject,
                        parent_norm=obj,
                        parent_name=obj,
                        source=source,
                    )
                    if result.single():
                        written += 1
                except Exception as e:
                    print(f"[Neo4jGraphStore] Error writing subdomain hierarchy {subject} -> {obj}: {e}")

        return written

    def write_subdomain_candidate(
        self,
        paper: Paper,
        assignment: SubdomainAssignment,
        source: str = "llm",
    ) -> None:
        """Persist a candidate subdomain proposal without modifying confirmed hierarchy."""
        candidate_name = self._normalize_subdomain_name(assignment.subdomain)
        parent_domain = self._normalize_subdomain_name(assignment.parent_domain or "biomedicine")

        with self.driver.session() as session:
            try:
                session.run(
                    """
                    MERGE (p:Paper {id: $paper_id})
                    ON CREATE SET p.created_at = timestamp()
                    SET p.updated_at = timestamp(),
                        p.title = $paper_title

                    MERGE (c:SubdomainCandidate {
                        normalized_name: $candidate_norm,
                        source_batch_id: $batch_id,
                        source_taxonomy_version: $taxonomy_version
                    })
                    ON CREATE SET c.created_at = timestamp()
                    SET c.updated_at = timestamp(),
                        c.name = $candidate_name,
                        c.suggested_parent = $suggested_parent,
                        c.confidence = $confidence,
                        c.reason = $reason,
                        c.status = 'candidate'

                    MERGE (p)-[r:SUGGESTS_SUBDOMAIN]->(c)
                    ON CREATE SET r.created_at = timestamp()
                    SET r.updated_at = timestamp(),
                        r.source = $source,
                        r.reason = $reason,
                        r.confidence = $confidence

                    WITH c
                    MATCH (parent:Subdomain {normalized_name: $suggested_parent})
                    MERGE (c)-[rp:CANDIDATE_SUBCLASS_OF]->(parent)
                    ON CREATE SET rp.created_at = timestamp()
                    SET rp.updated_at = timestamp(),
                        rp.source = $source
                    """,
                    paper_id=paper.id,
                    paper_title=paper.title,
                    candidate_norm=candidate_name,
                    candidate_name=candidate_name,
                    suggested_parent=parent_domain,
                    confidence=assignment.confidence,
                    reason=assignment.reason,
                    source=source,
                    batch_id=assignment.batch_id or "",
                    taxonomy_version=assignment.taxonomy_version,
                )
            except Exception as e:
                print(f"[Neo4jGraphStore] Error writing candidate subdomain for {paper.id}: {e}")

    def write_subdomain_assignment(
        self,
        paper: Paper,
        assignment: SubdomainAssignment,
        source: str = "llm",
    ) -> None:
        """Persist the paper, subdomain nodes, hierarchy, and paper classification edge."""
        self.write_paper_node(paper)

        subdomain = self._normalize_subdomain_name(assignment.subdomain)
        parent_domain = self._normalize_subdomain_name(assignment.parent_domain)

        if assignment.status == "candidate" or assignment.is_new_subdomain:
            self.write_subdomain_candidate(paper, assignment, source=source)
            return

        with self.driver.session() as session:
            try:
                session.run(
                    """
                    MERGE (p:Paper {id: $paper_id})
                    ON CREATE SET p.created_at = timestamp()
                    SET p.updated_at = timestamp(),
                        p.title = $paper_title,
                        p.subdomain = $subdomain_name,
                        p.parent_domain = $parent_domain,
                        p.subdomain_reason = $reason,
                        p.subdomain_confidence = $confidence

                    MERGE (s:Subdomain {normalized_name: $subdomain_norm})
                    ON CREATE SET s.created_at = timestamp()
                    SET s.updated_at = timestamp(),
                        s.name = $subdomain_name,
                        s.status = 'confirmed',
                        s.taxonomy_version = $taxonomy_version

                    MERGE (p)-[r:CLASSIFIED_AS]->(s)
                    ON CREATE SET r.created_at = timestamp()
                    SET r.updated_at = timestamp(),
                        r.source = $source,
                        r.reason = $reason,
                        r.confidence = $confidence
                    """,
                    paper_id=paper.id,
                    paper_title=paper.title,
                    subdomain_norm=subdomain,
                    subdomain_name=subdomain,
                    parent_domain=parent_domain,
                    reason=assignment.reason,
                    confidence=assignment.confidence,
                    taxonomy_version=assignment.taxonomy_version,
                    source=source,
                )
            except Exception as e:
                print(f"[Neo4jGraphStore] Error writing subdomain assignment for {paper.id}: {e}")

        self.write_subdomain_hierarchy(assignment.new_relations, source=source)

    def get_subdomain_hierarchy(self) -> List[Dict[str, Any]]:
        """Return current subdomain hierarchy edges stored in Neo4j."""
        with self.driver.session() as session:
            try:
                result = session.run(
                    """
                    MATCH (child:Subdomain)-[r:SUBCLASS_OF]->(parent:Subdomain)
                    RETURN child.name AS child,
                           child.normalized_name AS child_normalized,
                           parent.name AS parent,
                           parent.normalized_name AS parent_normalized,
                           r.source AS source
                    ORDER BY child.name, parent.name
                    """
                )
                return [dict(record) for record in result]
            except Exception as e:
                print(f"[Neo4jGraphStore] Error getting subdomain hierarchy: {e}")
                return []

    def get_current_taxonomy_version(self) -> int:
        """Return the current confirmed taxonomy version."""
        with self.driver.session() as session:
            try:
                result = session.run(
                    """
                    MATCH (s:Subdomain)
                    RETURN coalesce(max(s.taxonomy_version), 1) AS version
                    """
                )
                record = result.single()
                return int(record["version"]) if record and record["version"] is not None else 1
            except Exception as e:
                print(f"[Neo4jGraphStore] Error getting taxonomy version: {e}")
                return 1

    def get_confirmed_subdomains(self) -> List[Dict[str, Any]]:
        """Return confirmed subdomains with their direct parents."""
        with self.driver.session() as session:
            try:
                result = session.run(
                    """
                    MATCH (s:Subdomain)
                    OPTIONAL MATCH (s)-[:SUBCLASS_OF]->(p:Subdomain)
                    RETURN s.name AS name,
                           s.normalized_name AS normalized_name,
                           s.status AS status,
                           s.taxonomy_version AS taxonomy_version,
                           p.name AS parent_name,
                           p.normalized_name AS parent_normalized_name
                    ORDER BY s.name
                    """
                )
                return [dict(record) for record in result]
            except Exception as e:
                print(f"[Neo4jGraphStore] Error getting confirmed subdomains: {e}")
                return []

    def get_subdomain_candidates(self, batch_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return grouped candidate subdomains for refinement."""
        with self.driver.session() as session:
            try:
                query = """
                MATCH (c:SubdomainCandidate)
                WHERE c.status = 'candidate'
                """
                params: Dict[str, Any] = {}
                if batch_id:
                    query += "\nAND c.source_batch_id = $batch_id"
                    params["batch_id"] = batch_id

                query += """
                OPTIONAL MATCH (p:Paper)-[:SUGGESTS_SUBDOMAIN]->(c)
                RETURN c.name AS name,
                       c.normalized_name AS normalized_name,
                       c.suggested_parent AS suggested_parent,
                       c.source_taxonomy_version AS source_taxonomy_version,
                       collect(DISTINCT c.source_batch_id) AS batch_ids,
                       count(DISTINCT p.id) AS paper_count,
                       round(avg(c.confidence), 4) AS avg_confidence,
                       collect(DISTINCT p.id)[0..10] AS sample_papers
                ORDER BY paper_count DESC, avg_confidence DESC, name
                """
                result = session.run(query, **params)
                return [dict(record) for record in result]
            except Exception as e:
                print(f"[Neo4jGraphStore] Error getting subdomain candidates: {e}")
                return []

    def promote_candidate_subdomain(
        self,
        candidate_name: str,
        parent_name: str,
        taxonomy_version: int,
        source: str = "llm_refinement",
        reason: str = "",
    ) -> None:
        """Promote a candidate subdomain into the confirmed hierarchy."""
        candidate_norm = self._normalize_subdomain_name(candidate_name)
        parent_norm = self._normalize_subdomain_name(parent_name or "biomedicine")

        with self.driver.session() as session:
            try:
                session.run(
                    """
                    MATCH (c:SubdomainCandidate)
                    WHERE c.normalized_name = $candidate_norm
                      AND c.status = 'candidate'

                    MERGE (s:Subdomain {normalized_name: $candidate_norm})
                    ON CREATE SET s.created_at = timestamp()
                    SET s.updated_at = timestamp(),
                        s.name = $candidate_name,
                        s.status = 'confirmed',
                        s.taxonomy_version = $taxonomy_version

                    MERGE (p:Subdomain {normalized_name: $parent_norm})
                    ON CREATE SET p.created_at = timestamp()
                    SET p.updated_at = timestamp(),
                        p.name = $parent_name,
                        p.status = 'confirmed',
                        p.taxonomy_version = $taxonomy_version

                    MERGE (s)-[r:SUBCLASS_OF]->(p)
                    ON CREATE SET r.created_at = timestamp()
                    SET r.updated_at = timestamp(),
                        r.source = $source,
                        r.reason = $reason

                    WITH c, s
                    SET c.status = 'promoted',
                        c.updated_at = timestamp()
                    MERGE (c)-[m:PROMOTED_TO]->(s)
                    ON CREATE SET m.created_at = timestamp()
                    SET m.updated_at = timestamp(),
                        m.source = $source,
                        m.reason = $reason

                    WITH c, s
                    MATCH (paper:Paper)-[sg:SUGGESTS_SUBDOMAIN]->(c)
                    MERGE (paper)-[ca:CLASSIFIED_AS]->(s)
                    ON CREATE SET ca.created_at = timestamp()
                    SET ca.updated_at = timestamp(),
                        ca.source = $source,
                        ca.reason = $reason
                    SET paper.subdomain = s.name,
                        paper.parent_domain = $parent_name,
                        paper.subdomain_status = 'confirmed',
                        paper.taxonomy_version = $taxonomy_version
                    """,
                    candidate_norm=candidate_norm,
                    candidate_name=candidate_name,
                    parent_norm=parent_norm,
                    parent_name=parent_name,
                    taxonomy_version=taxonomy_version,
                    source=source,
                    reason=reason,
                )
            except Exception as e:
                print(f"[Neo4jGraphStore] Error promoting candidate {candidate_name}: {e}")

    def merge_candidate_into_subdomain(
        self,
        candidate_name: str,
        target_name: str,
        taxonomy_version: int,
        source: str = "llm_refinement",
        reason: str = "",
    ) -> None:
        """Merge a candidate subdomain into an existing confirmed subdomain."""
        candidate_norm = self._normalize_subdomain_name(candidate_name)
        target_norm = self._normalize_subdomain_name(target_name)

        with self.driver.session() as session:
            try:
                session.run(
                    """
                    MATCH (c:SubdomainCandidate)
                    WHERE c.normalized_name = $candidate_norm
                      AND c.status = 'candidate'

                    MERGE (s:Subdomain {normalized_name: $target_norm})
                    ON CREATE SET s.created_at = timestamp()
                    SET s.updated_at = timestamp(),
                        s.name = $target_name,
                        s.status = 'confirmed',
                        s.taxonomy_version = $taxonomy_version

                    WITH c, s
                    SET c.status = 'merged',
                        c.updated_at = timestamp()
                    MERGE (c)-[m:MERGED_INTO]->(s)
                    ON CREATE SET m.created_at = timestamp()
                    SET m.updated_at = timestamp(),
                        m.source = $source,
                        m.reason = $reason

                    WITH c, s
                    OPTIONAL MATCH (s)-[:SUBCLASS_OF]->(parent:Subdomain)
                    WITH c, s, parent
                    MATCH (paper:Paper)-[:SUGGESTS_SUBDOMAIN]->(c)
                    MERGE (paper)-[ca:CLASSIFIED_AS]->(s)
                    ON CREATE SET ca.created_at = timestamp()
                    SET ca.updated_at = timestamp(),
                        ca.source = $source,
                        ca.reason = $reason
                    SET paper.subdomain = s.name,
                        paper.parent_domain = coalesce(parent.name, paper.parent_domain),
                        paper.subdomain_status = 'confirmed',
                        paper.taxonomy_version = $taxonomy_version
                    """,
                    candidate_norm=candidate_norm,
                    target_norm=target_norm,
                    target_name=target_name,
                    taxonomy_version=taxonomy_version,
                    source=source,
                    reason=reason,
                )
            except Exception as e:
                print(f"[Neo4jGraphStore] Error merging candidate {candidate_name} into {target_name}: {e}")
    
    def write_claims(self, claims: List[KnowledgeClaim]) -> List[str]:
        """
        Write validated claims to the graph with ontology support.
        
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

                    # Final safety gate: never write self-loop relations
                    if claim.subject_id and claim.object_id and claim.subject_id == claim.object_id:
                        print(f"[Neo4jGraphStore] ✗ Skip self-loop by ID: {claim.subject} -[{claim.relation}]-> {claim.object}")
                        continue
                    if claim.subject.strip().lower() == claim.object.strip().lower():
                        print(f"[Neo4jGraphStore] ✗ Skip self-loop by name: {claim.subject} -[{claim.relation}]-> {claim.object}")
                        continue
                    
                    # Generate timestamp version
                    version = int(time.time() * 1000)
                    
                    # Sanitize relation type (现在支持中文)
                    rel_type = self._sanitize_relation_type(claim.relation)
                    
                    # Determine claim type and node labels
                    claim_type = self._determine_claim_type(claim)
                    
                    print(f"[Neo4jGraphStore] DEBUG: Claim: {claim.subject} [{claim.relation}] {claim.object} -> type={claim_type}")
                    
                    # Execute appropriate query based on claim type
                    record = self._execute_write_query(
                        session, claim, rel_type, version, claim_type
                    )
                    
                    if record:
                        claim_id = f"{claim.source_paper_id}_{record['subject_id']}_{record['object_id']}"
                        written_ids.append(claim_id)
                        
                        # Print with type indicator
                        type_indicator = {
                            "ontology_subclass": "🔷 本体-上下位",
                            "ontology_type": "🔶 本体-类型声明",
                            "ontology_property": "🟢 本体-属性约束",
                            "instance": "🔹 实例"
                        }.get(claim_type, "🔹 实例")
                        
                        print(f"[Neo4jGraphStore] ✓ {type_indicator}: {claim.subject} -[{rel_type}]-> {claim.object[:40]}...")
                    
                except Exception as e:
                    print(f"[Neo4jGraphStore] ✗ Error writing claim: {e}")
                    print(f"[Neo4jGraphStore]   Claim: {claim.subject} {claim.relation} {claim.object}")
        
        print(f"[Neo4jGraphStore] Successfully wrote {len(written_ids)} claims to graph")
        return written_ids
    
    def _determine_claim_type(self, claim: KnowledgeClaim) -> str:
        """
        Determine the type of claim for proper storage.
        
        Args:
            claim: KnowledgeClaim object
            
        Returns:
            Claim type string: 'ontology_subclass', 'ontology_type', 'ontology_property', 'instance'
        """
        relation = (claim.relation or "").strip().lower()

        if hasattr(claim, 'claim_type') and claim.claim_type == 'ontology':
            if relation in {"subclass_of", "subtype_of", "子类", "属于", "是", "为"}:
                return "ontology_subclass"
            elif relation in {"type", "instance_of", "is_a", "类型", "实例类型"}:
                return "ontology_type"
            else:
                return "ontology_property"
        
        # 根据关系词判断（扩展映射）
        ontology_relations = {
            # 上下位关系
            "子类": "ontology_subclass",
            "属于": "ontology_subclass",
            "是": "ontology_subclass",
            "为": "ontology_subclass",
            
            # 类型关系
            "类型": "ontology_type",
            "实例类型": "ontology_type",
            
            # 属性关系（中医特有）
            "藏": "ontology_property",
            "主": "ontology_property",
            "生": "ontology_property",
            "克": "ontology_property",
            "合": "ontology_property",
            "开窍于": "ontology_property",
            "其华在": "ontology_property",
            "走": "ontology_property",
            "入": "ontology_property",
            "伤": "ontology_property",
            "胜": "ontology_property",
            "畏": "ontology_property",
            "恶": "ontology_property",
            "苦": "ontology_property",
            "欲": "ontology_property",
            "出": "ontology_property",
            "起于": "ontology_property",
            "络": "ontology_property",
            "循": "ontology_property",
            "属": "ontology_property",
            "应": "ontology_property",
            "通于": "ontology_property"
        }
        
        if claim.relation in ontology_relations:
            return ontology_relations[claim.relation]

        english_ontology_relations = {
            "subclass_of": "ontology_subclass",
            "subtype_of": "ontology_subclass",
            "type": "ontology_type",
            "instance_of": "ontology_type",
            "is_a": "ontology_type",
            "biomarker_for": "ontology_property",
            "risk_factor_for": "ontology_property",
            "associated_with": "ontology_property",
            "treats": "ontology_property",
            "measured_in": "ontology_property",
            "part_of": "ontology_property",
            "inhibits": "ontology_property",
            "activates": "ontology_property",
            "causes": "ontology_property",
            "predicts": "ontology_property",
        }
        if relation in english_ontology_relations:
            return english_ontology_relations[relation]
        
        # 检查实体名是否为抽象概念（辅助判断）
        abstract_concepts = ["心", "肝", "脾", "肺", "肾", "胃", "胆", "大肠", "小肠", 
                            "膀胱", "三焦", "精", "气", "血", "津", "液", "神", "魂", "魄",
                            "意", "志", "风", "寒", "暑", "湿", "燥", "火", "阴阳", "五行",
                            "木", "火", "土", "金", "水", "酸", "苦", "甘", "辛", "咸",
                            "春", "夏", "秋", "冬", "长夏"]
        
        if claim.subject in abstract_concepts and claim.object in abstract_concepts:
            return "ontology_property"
        
        # 默认作为实例
        return "instance"
    
    def _execute_write_query(self, session, claim: KnowledgeClaim, rel_type: str, 
                             version: int, claim_type: str) -> Optional[Dict]:
        """
        Execute the appropriate Cypher query based on claim type.
        """
        # 为本体节点添加额外的标签和属性
        subject_labels = "Entity"
        object_labels = "Entity"
        
        if claim_type in ["ontology_subclass", "ontology_type", "ontology_property"]:
            subject_labels = "Entity:Concept"
            object_labels = "Entity:Concept"
        
        # Case 1: 上下位关系
        if claim_type == "ontology_subclass":
            query = f"""
            MERGE (s:{subject_labels} {{id: $subject_id}})
            ON CREATE SET s.name = $subject_name, 
                         s.created_at = timestamp(),
                         s.is_concept = true
            ON MATCH SET s.updated_at = timestamp(),
                         s.is_concept = true
            
            MERGE (o:{object_labels} {{id: $object_id}})
            ON CREATE SET o.name = $object_name, 
                         o.created_at = timestamp(),
                         o.is_concept = true
            ON MATCH SET o.updated_at = timestamp(),
                         o.is_concept = true
            
            CREATE (s)-[r:SUBCLASS_OF]->(o)
            SET r.version = $version,
                r.source_paper = $source_paper,
                r.evidence = $evidence,
                r.created_at = timestamp(),
                r.relation_original = $relation_original,
                r.is_ontology = true
            
            RETURN s.id as subject_id, o.id as object_id, type(r) as rel_type
            """
            
        # Case 2: 实例类型关系
        elif claim_type == "ontology_type":
            query = f"""
            MERGE (s:Entity {{id: $subject_id}})
            ON CREATE SET s.name = $subject_name, 
                         s.created_at = timestamp()
            ON MATCH SET s.updated_at = timestamp()
            
            MERGE (o:{object_labels} {{id: $object_id}})
            ON CREATE SET o.name = $object_name, 
                         o.created_at = timestamp(),
                         o.is_concept = true
            ON MATCH SET o.updated_at = timestamp(),
                         o.is_concept = true
            
            CREATE (s)-[r:TYPE]->(o)
            SET r.version = $version,
                r.source_paper = $source_paper,
                r.evidence = $evidence,
                r.created_at = timestamp(),
                r.relation_original = $relation_original,
                r.is_ontology = true
            
            RETURN s.id as subject_id, o.id as object_id, type(r) as rel_type
            """
            
        # Case 3: 本体属性关系
        elif claim_type == "ontology_property":
            query = f"""
            MERGE (s:{subject_labels} {{id: $subject_id}})
            ON CREATE SET s.name = $subject_name, 
                         s.created_at = timestamp(),
                         s.is_concept = true
            ON MATCH SET s.updated_at = timestamp(),
                         s.is_concept = true
            
            MERGE (o:{object_labels} {{id: $object_id}})
            ON CREATE SET o.name = $object_name, 
                         o.created_at = timestamp(),
                         o.is_concept = true
            ON MATCH SET o.updated_at = timestamp(),
                         o.is_concept = true
            
            CREATE (s)-[r:{rel_type}]->(o)
            SET r.version = $version,
                r.source_paper = $source_paper,
                r.evidence = $evidence,
                r.created_at = timestamp(),
                r.relation_original = $relation_original,
                r.is_ontology = true
            
            RETURN s.id as subject_id, o.id as object_id, type(r) as rel_type
            """
                
        # Case 4: 实例关系
        else:
            query = f"""
            MERGE (s:Entity {{id: $subject_id}})
            ON CREATE SET s.name = $subject_name, 
                         s.created_at = timestamp()
            ON MATCH SET s.updated_at = timestamp()
            
            MERGE (o:Entity {{id: $object_id}})
            ON CREATE SET o.name = $object_name, 
                         o.created_at = timestamp()
            ON MATCH SET o.updated_at = timestamp()
            
            CREATE (s)-[r:{rel_type}]->(o)
            SET r.version = $version,
                r.source_paper = $source_paper,
                r.evidence = $evidence,
                r.created_at = timestamp(),
                r.relation_original = $relation_original
            
            RETURN s.id as subject_id, o.id as object_id, type(r) as rel_type
            """
        
        try:
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
            return result.single()
        except Exception as e:
            print(f"[Neo4jGraphStore] Query execution error: {e}")
            return None
    
    def query(self, cypher_query: str, **kwargs) -> List[Dict[str, Any]]:
        """Execute a Cypher query on the graph database."""
        results = []
        with self.driver.session() as session:
            try:
                result = session.run(cypher_query, **kwargs)
                results = [dict(record) for record in result]
            except Exception as e:
                print(f"[Neo4jGraphStore] Query error: {e}")
        return results
    
    def clear_all(self):
        """Clear all nodes and relationships from the database."""
        with self.driver.session() as session:
            try:
                session.run("MATCH (n) DETACH DELETE n")
                print("[Neo4jGraphStore] All data cleared from database")
            except Exception as e:
                print(f"[Neo4jGraphStore] Error clearing database: {e}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get database statistics with ontology separation."""
        stats = {
            "nodes": 0, "relationships": 0, "concept_nodes": 0,
            "entity_nodes": 0, "ontology_relations": 0, "instance_relations": 0
        }
        with self.driver.session() as session:
            try:
                result = session.run("MATCH (n) RETURN count(n) as count")
                stats["nodes"] = result.single()["count"]
                
                result = session.run("""
                    MATCH (n) 
                    WHERE n.is_concept = true OR 'Concept' IN labels(n)
                    RETURN count(n) as count
                """)
                stats["concept_nodes"] = result.single()["count"]
                
                # 修改点：EXISTS 替换为 IS NULL
                result = session.run("""
                    MATCH (n) 
                    WHERE (n.is_concept <> true AND NOT 'Concept' IN labels(n)) 
                       OR n.is_concept IS NULL
                    RETURN count(n) as count
                """)
                stats["entity_nodes"] = result.single()["count"]
                
                result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                stats["relationships"] = result.single()["count"]
                
                # 统计本体关系（使用中文关系类型）
                result = session.run("""
                    MATCH ()-[r]->() 
                    WHERE r.is_ontology = true 
                       OR type(r) IN ['SUBCLASS_OF', 'TYPE', '子类', '类型']
                    RETURN count(r) as count
                """)
                stats["ontology_relations"] = result.single()["count"]
                
                stats["instance_relations"] = stats["relationships"] - stats["ontology_relations"]
                print(f"[Neo4jGraphStore] Stats: "
                      f"概念={stats['concept_nodes']}, 实体={stats['entity_nodes']}, "
                      f"本体关系={stats['ontology_relations']}, 实例关系={stats['instance_relations']}")
            except Exception as e:
                print(f"[Neo4jGraphStore] Error getting stats: {e}")
        return stats
    
    def get_ontology_hierarchy(self, root_concept: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get the ontology hierarchy tree."""
        with self.driver.session() as session:
            try:
                if root_concept:
                    query = """
                    MATCH (s:Concept)-[r]->(o:Concept)
                    WHERE type(r) IN ['SUBCLASS_OF', '子类']
                      AND (s.name = $root OR o.name = $root)
                    RETURN s.name as subject, o.name as object, r.evidence as evidence
                    """
                    result = session.run(query, root=root_concept)
                else:
                    query = """
                    MATCH (s)-[r]->(o)
                    WHERE (s.is_concept = true OR 'Concept' IN labels(s))
                      AND (o.is_concept = true OR 'Concept' IN labels(o))
                      AND type(r) IN ['SUBCLASS_OF', '子类']
                    RETURN s.name as subject, o.name as object, r.evidence as evidence
                    LIMIT 100
                    """
                    result = session.run(query)
                return [dict(record) for record in result]
            except Exception as e:
                print(f"[Neo4jGraphStore] Error getting ontology hierarchy: {e}")
                return []
    
    def get_instances_of_concept(self, concept_name: str) -> List[str]:
        """Get all instances belonging to a concept."""
        with self.driver.session() as session:
            try:
                query = """
                MATCH (e:Entity)-[r]->(c)
                WHERE c.name = $concept OR (c.is_concept = true AND c.name = $concept)
                  AND type(r) IN ['TYPE', '类型']
                RETURN e.name as instance_name
                """
                result = session.run(query, concept=concept_name)
                return [record["instance_name"] for record in result]
            except Exception as e:
                print(f"[Neo4jGraphStore] Error getting instances: {e}")
                return []

    # Graph Cleanup Methods (保持不变)
    def run_garbage_collection(self, preserve_ontology: bool = True) -> int:
        """Delete isolated nodes (Entity nodes with no relationships)."""
        with self.driver.session() as session:
            try:
                if preserve_ontology:
                    query = """
                    MATCH (n:Entity)
                    WHERE NOT (n)--() 
                      AND (n.is_concept <> true AND NOT 'Concept' IN labels(n))
                    DELETE n
                    RETURN count(*) as deleted_count
                    """
                else:
                    query = """
                    MATCH (n)
                    WHERE NOT (n)--()
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
        """Merge two entities."""
        with self.driver.session() as session:
            try:
                check_query = """
                MATCH (p {id: $primary_id})
                MATCH (s {id: $secondary_id})
                RETURN p.id as primary, labels(p)[0] as p_type, p.is_concept as p_concept,
                       s.id as secondary, labels(s)[0] as s_type, s.is_concept as s_concept
                """
                result = session.run(check_query, primary_id=primary_id, secondary_id=secondary_id)
                record = result.single()
                if not record:
                    return False
                if record["p_concept"] != record["s_concept"]:
                    print(f"[Neo4jGraphStore] Cannot merge concept with non-concept")
                    return False
                self._manual_merge_relationships(session, primary_id, secondary_id)
                delete_query = """
                MATCH (secondary {id: $secondary_id})
                DETACH DELETE secondary
                """
                session.run(delete_query, secondary_id=secondary_id)
                return True
            except Exception as e:
                print(f"[Neo4jGraphStore] Error merging entities: {e}")
                return False
    
    def _manual_merge_relationships(self, session, primary_id: str, secondary_id: str):
        """Manually transfer relationships when APOC is not available."""
        get_rels_query = """
        MATCH (secondary {id: $secondary_id})-[r]-(other)
        WHERE other.id <> $primary_id AND other.id <> $secondary_id
        RETURN type(r) as rel_type, startNode(r) = secondary as is_outgoing,
               other.id as other_id, labels(other)[0] as other_label, properties(r) as props
        """
        result = session.run(get_rels_query, secondary_id=secondary_id, primary_id=primary_id)
        for record in result:
            rel_type = record["rel_type"]
            is_outgoing = record["is_outgoing"]
            other_id = record["other_id"]
            other_label = record["other_label"]
            props = dict(record["props"]) if record["props"] else {}
            try:
                if is_outgoing:
                    create_query = f"""
                    MATCH (primary {{id: $primary_id}}), (other:{other_label} {{id: $other_id}})
                    WHERE NOT (primary)-[:{rel_type}]->(other)
                    CREATE (primary)-[r:{rel_type}]->(other)
                    SET r = $props
                    """
                else:
                    create_query = f"""
                    MATCH (primary {{id: $primary_id}}), (other:{other_label} {{id: $other_id}})
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
                    MATCH (e)
                    WITH toLower(e.name) as lower_name, collect({
                        id: e.id, name: e.name, type: labels(e)[0],
                        entity_type: e.entity_type, is_concept: e.is_concept,
                        created_at: e.created_at
                    }) as entities
                    WHERE size(entities) > 1
                    RETURN lower_name as group_name, entities, size(entities) as count
                    ORDER BY count DESC
                    """
                else:
                    query = """
                    MATCH (e)
                    WITH e.name as name, collect({
                        id: e.id, name: e.name, type: labels(e)[0],
                        entity_type: e.entity_type, is_concept: e.is_concept,
                        created_at: e.created_at
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
                return groups
            except Exception as e:
                print(f"[Neo4jGraphStore] Error finding duplicate entities: {e}")
                return []
    
    def get_all_ontology_relations(self) -> List[Dict[str, Any]]:
        """获取所有本体关系（包括显式和推断的）"""
        with self.driver.session() as session:
            try:
                query = """
                MATCH (s)-[r]->(o)
                WHERE r.is_ontology = true 
                   OR type(r) IN ['子类', '类型']
                   OR (s.is_concept = true AND o.is_concept = true)
                RETURN s.id as subject_id, s.name as subject_name, labels(s) as subject_labels,
                       type(r) as relation_type, o.id as object_id, o.name as object_name,
                       labels(o) as object_labels, r.evidence as evidence,
                       r.source_paper as source_paper, r.is_ontology as is_ontology
                ORDER BY s.name, o.name
                """
                result = session.run(query)
                return [dict(record) for record in result]
            except Exception as e:
                print(f"[Neo4jGraphStore] Error getting ontology relations: {e}")
                return []
    
    def get_all_concepts(self) -> List[Dict[str, Any]]:
        """获取所有概念节点"""
        with self.driver.session() as session:
            try:
                query = """
                MATCH (n)
                WHERE n.is_concept = true OR 'Concept' IN labels(n)
                RETURN n.id as id, n.name as name, labels(n) as labels,
                       n.entity_type as entity_type, n.created_at as created_at
                ORDER BY n.name
                """
                result = session.run(query)
                return [dict(record) for record in result]
            except Exception as e:
                print(f"[Neo4jGraphStore] Error getting concepts: {e}")
                return []
