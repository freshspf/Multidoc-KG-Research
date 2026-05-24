"""
Semantic Grounding Agent: Aligns entities with existing graph using Vector Search + LLM Judge.
"""
import json
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict
from schema import KnowledgeClaim, ClaimStatus
from core.llm_client import LLMClient
from core.vector_store import VectorStore


class SemanticGroundingAgent:
    """Agent responsible for grounding entities to existing graph nodes."""

    def __init__(self, llm_client: LLMClient, vector_store: VectorStore):
        """
        Initialize Semantic Grounding Agent.

        Args:
            llm_client: LLM client for entity grounding judgment
            vector_store: Vector store for semantic entity search
        """
        self.llm_client = llm_client
        self.vector_store = vector_store
        self._entity_cache: Dict[str, str] = {}  # 方案1: 实体解析缓存
        self._cache_lock = threading.Lock()       # 线程安全锁（配合方案2）
        print("[SemanticGroundingAgent] Initialized")
    
    def _resolve_entity(self, entity_name: str, context: str = "") -> str:
        """
        Resolve an entity to a grounded ID using Vector Search + LLM Judge.
        
        Strategy:
        1. Retrieval: Search vector store for similar entities
        2. Exact Match: If exact match found (case-insensitive), return existing ID
        3. LLM Judge: If candidates exist, use LLM to decide merge vs. new
        4. Create New: If no match, generate new ID and add to vector store
        
        Args:
            entity_name: Name of the entity to resolve
            context: Optional context about the entity
            
        Returns:
            Grounded entity ID (existing or newly created)
        """
        # 方案1: 命中缓存直接返回，跳过 FAISS + LLM
        cache_key = entity_name.lower()
        with self._cache_lock:
            if cache_key in self._entity_cache:
                print(f"[Grounding] Cache hit: '{entity_name}' -> '{self._entity_cache[cache_key]}'")
                return self._entity_cache[cache_key]

        # Step 1: Retrieval - Search for similar entities
        candidates = self.vector_store.search(entity_name, top_k=5)

        # Step 2: Check for exact match (case-insensitive)
        for candidate in candidates:
            if candidate['name'].lower() == entity_name.lower():
                print(f"[Grounding] Exact match found: '{entity_name}' -> '{candidate['name']}' (ID: {candidate['id']})")
                with self._cache_lock:
                    self._entity_cache[cache_key] = candidate['id']
                return candidate['id']

        # Step 3: Filter candidates by score threshold (L2 distance < 1.0 means quite similar)
        # For sentence embeddings, L2 distance typically ranges from 0 to ~2
        score_threshold = 1.0
        good_candidates = [c for c in candidates if c['score'] < score_threshold]

        # Step 4: If no good candidates, create new entity
        if not good_candidates:
            new_id = f"ent_{uuid.uuid4().hex[:8]}"
            self.vector_store.add_entity(entity_name, new_id)
            print(f"[Grounding] Created New Entity: '{entity_name}' (ID: {new_id})")
            with self._cache_lock:
                self._entity_cache[cache_key] = new_id
            return new_id

        # Step 5: Use LLM Judge to decide
        decision = self._llm_judge_entity(entity_name, good_candidates, context)

        if decision['decision'] == 'merge' and 'id' in decision:
            resolved_id = decision['id']
            merged_candidate = next((c for c in good_candidates if c['id'] == resolved_id), None)
            if merged_candidate:
                print(f"[Grounding] Merged: '{entity_name}' -> '{merged_candidate['name']}' (ID: {resolved_id})")
            else:
                print(f"[Grounding] Merged: '{entity_name}' -> (ID: {resolved_id})")
        else:
            resolved_id = f"ent_{uuid.uuid4().hex[:8]}"
            self.vector_store.add_entity(entity_name, resolved_id)
            print(f"[Grounding] Created New Entity: '{entity_name}' (ID: {resolved_id}) [LLM decided not to merge]")

        with self._cache_lock:
            self._entity_cache[cache_key] = resolved_id
        return resolved_id
    
    def _llm_judge_entity(self, target_entity: str, candidates: List[Dict], context: str = "") -> Dict[str, str]:
        """
        Use LLM to judge whether to merge target entity with existing candidates.
        
        Args:
            target_entity: The new entity to ground
            candidates: List of candidate entities from vector search
            context: Optional context about the entity
            
        Returns:
            Dictionary with 'decision' ('merge' or 'new') and optionally 'id'
        """
        # Build candidate list for prompt
        candidate_text = "\n".join([
            f"  - '{c['name']}' (ID: {c['id']}, similarity score: {c['score']:.3f})"
            for c in candidates
        ])
        
        system_prompt = (
            "You are a Knowledge Graph Entity Resolver. "
            "Your task is to determine if a target entity should be merged with an existing entity or created as new. "
            "Output ONLY valid JSON."
        )
        
        user_prompt = (
            f"Target Entity: '{target_entity}'\n\n"
            f"Candidate Entities:\n{candidate_text}\n\n"
        )
        
        if context:
            user_prompt += f"Context: {context}\n\n"
        
        user_prompt += (
            "Question: Is the target entity the same as any of these candidates?\n\n"
            "Rules:\n"
            "- Consider semantic meaning, not just text similarity\n"
            "- Abbreviations should match full forms (e.g., 'CoT' = 'Chain-of-Thought')\n"
            "- Different concepts should NOT be merged even if similar\n\n"
            "Respond with JSON:\n"
            "- If SAME: {\"decision\": \"merge\", \"id\": \"<candidate_id>\", \"reasoning\": \"...\"}\n"
            "- If DIFFERENT: {\"decision\": \"new\", \"reasoning\": \"...\"}\n\n"
            "Return only valid JSON, no markdown formatting."
        )
        
        try:
            response_text = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True
            )
            
            print(f"[Grounding] LLM Judge response for '{target_entity}': {response_text[:200]}...")
            
            # Clean response
            if isinstance(response_text, list):
                response_text = json.dumps(response_text)
            elif not isinstance(response_text, str):
                response_text = str(response_text)
            
            response_clean = response_text.strip()
            if response_clean.startswith("```json"):
                response_clean = response_clean[7:]
            elif response_clean.startswith("```"):
                response_clean = response_clean[3:]
            if response_clean.endswith("```"):
                response_clean = response_clean[:-3]
            response_clean = response_clean.strip()
            
            # Parse JSON
            try:
                parsed = json.loads(response_clean)
                
                # Handle list response
                if isinstance(parsed, list):
                    if len(parsed) > 0 and isinstance(parsed[0], dict):
                        decision = parsed[0]
                    else:
                        print(f"[Grounding] Warning: Empty or invalid list response")
                        return {"decision": "new", "reasoning": "Invalid LLM response"}
                elif isinstance(parsed, dict):
                    decision = parsed
                else:
                    print(f"[Grounding] Warning: Unexpected response type {type(parsed)}")
                    return {"decision": "new", "reasoning": "Invalid response type"}
                
                # Validate decision
                if decision.get("decision") not in ["merge", "new"]:
                    print(f"[Grounding] Warning: Invalid decision value: {decision.get('decision')}")
                    return {"decision": "new", "reasoning": "Invalid decision format"}
                
                return decision
                
            except json.JSONDecodeError as e:
                print(f"[Grounding] Warning: Failed to parse JSON from LLM: {str(e)}")
                print(f"[Grounding] Response preview: {response_clean[:200]}...")
                return {"decision": "new", "reasoning": f"JSON parsing error: {str(e)}"}
                
        except Exception as e:
            print(f"[Grounding] Error in LLM judge: {str(e)}")
            return {"decision": "new", "reasoning": f"LLM error: {str(e)}"}
    
    def process(self, claims: List[KnowledgeClaim]) -> List[KnowledgeClaim]:
        """
        Ground entities in claims to existing graph nodes.
        
        Args:
            claims: List of KnowledgeClaim objects to ground
            
        Returns:
            List of grounded KnowledgeClaim objects with updated entity IDs
        """
        print(f"[SemanticGroundingAgent] Processing {len(claims)} claims for grounding")
        
        grounded_claims = []

        def _ground_claim(claim):
            try:
                context = f"Evidence: {claim.evidence[:200]}..."
                # 方案2: 并行解析 subject 和 object
                with ThreadPoolExecutor(max_workers=2) as ex:
                    fut_subj = ex.submit(self._resolve_entity, claim.subject, context)
                    fut_obj  = ex.submit(self._resolve_entity, claim.object,  context)
                    claim.subject_id = fut_subj.result()
                    claim.object_id  = fut_obj.result()
                claim.grounded_ids = [claim.subject_id, claim.object_id]
                claim.status = ClaimStatus.GROUNDED
            except Exception as e:
                print(f"[Grounding] Error grounding claim: {str(e)}")
                claim.status = ClaimStatus.EXTRACTED
            return claim

        for claim in claims:
            grounded_claims.append(_ground_claim(claim))
        
        print(f"[SemanticGroundingAgent] Grounded {len(grounded_claims)} claims")
        return grounded_claims
