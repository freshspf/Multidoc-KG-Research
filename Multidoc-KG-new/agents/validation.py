"""
Knowledge Validation Agent: Checks for logical conflicts with history using LLM Judge.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from typing import List
from schema import KnowledgeClaim, ClaimStatus
from core.llm_client import LLMClient
from core.graph_store import MockGraphStore


class KnowledgeValidationAgent:
    """Agent responsible for validating claims against existing graph knowledge."""
    
    def __init__(self, llm_client: LLMClient, graph_store: MockGraphStore):
        """
        Initialize Knowledge Validation Agent.
        
        Args:
            llm_client: LLM client for validation judgment
            graph_store: Graph database store for querying historical claims
        """
        self.llm_client = llm_client
        self.graph_store = graph_store
        print("[KnowledgeValidationAgent] Initialized")
    
    def process(self, claims: List[KnowledgeClaim]) -> List[KnowledgeClaim]:
        """
        Validate claims for logical conflicts with existing graph.
        
        Args:
            claims: List of grounded KnowledgeClaim objects to validate
            
        Returns:
            List of validated or rejected KnowledgeClaim objects
        """
        print(f"[KnowledgeValidationAgent] Processing {len(claims)} claims for validation")
        
        # TODO: Query graph_store for related historical claims
        # TODO: For each claim, use LLM judge to check for conflicts
        # TODO: Update claim status to VALIDATED or REJECTED based on judgment
        
        def _validate_claim(claim):
            try:
                # Build claim string
                claim_str = f"{claim.subject} {claim.relation} {claim.object}"
                
                # Retrieve historical context for the subject entity
                # Use subject name for context retrieval (more meaningful than random ID)
                historical_context = []
                if claim.subject:
                    historical_context = self.graph_store.get_entity_context(claim.subject)
                
                # Build context-aware prompt
                system_prompt = (
                    "You are a knowledge validation judge. "
                    "Your task is to detect logical conflicts between new claims and existing knowledge. "
                    "Output ONLY valid JSON."
                )
                
                user_prompt = f"New Claim: {claim_str}\n"
                user_prompt += f"Evidence: {claim.evidence[:200]}...\n\n"
                
                if historical_context:
                    context_str = "\n".join([f"  - {ctx}" for ctx in historical_context])
                    user_prompt += f"Existing Knowledge in Graph:\n{context_str}\n\n"
                    user_prompt += (
                        "Task: Check for logical conflicts between the New Claim and Existing Knowledge.\n\n"
                        "Rules:\n"
                        "- If the New Claim CONTRADICTS Existing Knowledge, return {\"valid\": false, \"type\": \"conflict\", \"reasoning\": \"...\"}\n"
                        "- If the New Claim SUPPORTS or is CONSISTENT with Existing Knowledge, return {\"valid\": true, \"type\": \"support\", \"reasoning\": \"...\"}\n"
                        "- Consider semantic meaning and context, not just surface-level text similarity\n\n"
                    )
                else:
                    user_prompt += "Existing Knowledge: No historical context found for this entity.\n\n"
                    user_prompt += (
                        "Task: Evaluate if the claim is logically consistent and well-formed.\n\n"
                        "Since there's no existing knowledge, return {\"valid\": true, \"type\": \"new\", \"reasoning\": \"...\"} if the claim is valid.\n\n"
                    )
                
                user_prompt += (
                    "Response Format (JSON only, no markdown):\n"
                    "{\n"
                    "  \"valid\": boolean,\n"
                    "  \"type\": \"support\" | \"conflict\" | \"new\",\n"
                    "  \"confidence\": float (0-1),\n"
                    "  \"reasoning\": string\n"
                    "}"
                )
                
                print(f"[KnowledgeValidationAgent] Validating claim: '{claim_str[:60]}...'")
                if historical_context:
                    print(f"[KnowledgeValidationAgent] Found {len(historical_context)} historical claims for conflict check")
                
                response_text = self.llm_client.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    json_mode=True
                )
                
                print(f"[KnowledgeValidationAgent] LLM response preview: {response_text[:150]}...")
                
                # Clean response - remove markdown code blocks if present
                if isinstance(response_text, list):
                    print(f"[KnowledgeValidationAgent] Warning: Response is a list, converting to JSON string")
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
                
                # Parse JSON response
                try:
                    parsed = json.loads(response_clean)
                    
                    # Handle list response - take first item if list
                    if isinstance(parsed, list):
                        if len(parsed) > 0:
                            judgment = parsed[0] if isinstance(parsed[0], dict) else {}
                        else:
                            print(f"[KnowledgeValidationAgent] Warning: Empty list response, marking as disputed")
                            judgment = {"valid": False, "confidence": 0.0, "reasoning": "Empty response from LLM"}
                    elif isinstance(parsed, dict):
                        judgment = parsed
                    else:
                        print(f"[KnowledgeValidationAgent] Warning: Unexpected response type {type(parsed)}, marking as disputed")
                        judgment = {"valid": False, "confidence": 0.0, "reasoning": f"Unexpected response type: {type(parsed)}"}
                    
                except json.JSONDecodeError as e:
                    print(f"[KnowledgeValidationAgent] Warning: Failed to parse JSON response: {str(e)}")
                    print(f"[KnowledgeValidationAgent] Response preview: {response_clean[:200]}...")
                    judgment = {"valid": False, "confidence": 0.0, "reasoning": f"JSON parsing error: {str(e)}"}
                
                # Extract validation type
                validation_type = judgment.get("type", "unknown")
                claim.validation_type = validation_type
                
                # Process judgment
                is_valid = judgment.get("valid", False)
                confidence = judgment.get("confidence", 0.0)
                reasoning = judgment.get("reasoning", "")
                
                if isinstance(judgment, dict) and is_valid:
                    claim.status = ClaimStatus.VALIDATED
                    print(f"[KnowledgeValidationAgent] ✓ VALIDATED: Type={validation_type}, Confidence={confidence:.2f}")
                else:
                    claim.status = ClaimStatus.REJECTED
                    print(f"[KnowledgeValidationAgent] ✗ REJECTED: Type={validation_type}, Reason: {reasoning[:80]}...")

            except Exception as e:
                print(f"[KnowledgeValidationAgent] Error validating claim: {str(e)}")
                claim.status = ClaimStatus.REJECTED
            return claim

        # 并行验证所有 claims
        with ThreadPoolExecutor(max_workers=8) as ex:
            validated_claims = list(ex.map(_validate_claim, claims))
        
        validated_count = sum(1 for c in validated_claims if c.status == ClaimStatus.VALIDATED)
        rejected_count = len(validated_claims) - validated_count
        print(f"[KnowledgeValidationAgent] Validated {validated_count} claims, rejected {rejected_count}")
        
        return validated_claims
