#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTL File Evaluator
Use GPT-4o to evaluate knowledge graphs extracted from TTL files
"""

import os
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import openai
from dataclasses import dataclass
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Evaluation result data class"""
    meta: Dict[str, Any]
    scores: Dict[str, Dict[str, Any]]
    final_score: float
    summary_advice: str
    top_fixes: List[str]
    passed_threshold: bool


class TTLEvaluator:
    """TTL File Evaluator"""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, threshold: float = 7, config: Dict = None):
        """
        Initialize evaluator

        Args:
            api_key: OpenAI API key, if None will get from environment variable OPENAI_API_KEY
            base_url: OpenAI base URL, if None will get from environment variable OPENAI_BASE_URL
            threshold: Score threshold, default 8
            config: Configuration dictionary
        """
        # Load configuration
        self.config = config or {}
        evaluator_config = self.config.get('evaluator', {})
        api_config = self.config.get('api', {})

        # Model configuration
        self.model = evaluator_config.get('model', 'gpt-4o-mini')
        self.temperature = evaluator_config.get('temperature', 0.1)
        self.max_tokens = evaluator_config.get('max_tokens', 2000)
        self.threshold = evaluator_config.get('threshold', threshold)

        # Get API key and base URL from config first, then environment variables, then parameters
        api_key = api_config.get('openai_api_key') or api_key or os.getenv('OPENAI_API_KEY')
        base_url = api_config.get('openai_base_url') or base_url or os.getenv('OPENAI_BASE_URL')

        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")

        # Initialize OpenAI client
        api_config = self.config.get('api', {})
        timeout = api_config.get('timeout', 600.0)  # 默认 10 分钟

        client_kwargs = {"api_key": api_key, "timeout": timeout}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = openai.OpenAI(**client_kwargs)

        # Load evaluation prompt template
        self.prompt_template = self._load_prompt_template()
    
    def _load_prompt_template(self) -> str:
        """Load evaluation prompt template"""
        prompt_file = Path(__file__).parent / "evaluator_prompt.txt"
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {prompt_file}")
            raise
    
    def _extract_ttl_content(self, ttl_file_path: str) -> str:
        """
        Extract content from TTL file
        
        Args:
            ttl_file_path: TTL file path
            
        Returns:
            Extracted TTL content string
        """
        try:
            with open(ttl_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract turtle block content
            turtle_blocks = []
            in_turtle_block = False
            current_block = []
            
            for line in content.split('\n'):
                if line.strip() == '```turtle':
                    in_turtle_block = True
                    current_block = []
                elif line.strip() == '```' and in_turtle_block:
                    in_turtle_block = False
                    if current_block:
                        turtle_blocks.append('\n'.join(current_block))
                elif in_turtle_block:
                    current_block.append(line)
            
            # Combine all turtle blocks
            combined_ttl = '\n\n'.join(turtle_blocks)
            
            logger.info(f"Extracted {len(turtle_blocks)} turtle blocks from file {ttl_file_path}")
            return combined_ttl
            
        except FileNotFoundError:
            logger.error(f"TTL file not found: {ttl_file_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading TTL file: {e}")
            raise
    
    def _count_triples_and_entities(self, ttl_content: str) -> Tuple[int, int]:
        """
        Count triples and entities
        
        Args:
            ttl_content: TTL content
            
        Returns:
            (triple count, entity count)
        """
        # Simple triple counting (lines containing ';' or '.')
        lines = ttl_content.split('\n')
        triple_count = 0
        entities = set()
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                if line.endswith('.') or line.endswith(';'):
                    triple_count += 1
                
                # Extract entities (simplified version)
                if ':' in line and ('rdf:type' in line or 'a ' in line):
                    parts = line.split()
                    if parts:
                        entity = parts[0].rstrip()
                        if entity.startswith(':'):
                            entities.add(entity)
        
        return triple_count, len(entities)
    
    def _extract_sections(self, ttl_content: str) -> List[str]:
        """
        Extract sections from TTL content
        
        Args:
            ttl_content: TTL content
            
        Returns:
            List of sections
        """
        sections = set()
        
        # Find sourceSection properties
        section_pattern = r':sourceSection\s+"([^"]+)"'
        matches = re.findall(section_pattern, ttl_content)
        sections.update(matches)
        
        # Find section information in comments
        comment_pattern = r'# ===== SECTION: ([^=]+) ====='
        comment_matches = re.findall(comment_pattern, ttl_content)
        sections.update([s.strip() for s in comment_matches])
        
        return list(sections)
    
    def evaluate_ttl(self, ttl_file_path: str) -> EvaluationResult:
        """
        Evaluate TTL file comprehensively
        
        Args:
            ttl_file_path: TTL file path
            
        Returns:
            Evaluation result
        """
        logger.info(f"Starting comprehensive TTL file evaluation: {ttl_file_path}")
        
        # Extract TTL content
        ttl_content = self._extract_ttl_content(ttl_file_path)
        
        if not ttl_content.strip():
            raise ValueError("TTL file content is empty or invalid")
        
        # Count basic information
        triple_count, entity_count = self._count_triples_and_entities(ttl_content)
        sections = self._extract_sections(ttl_content)
        
        logger.info(f"Found {triple_count} triples, {entity_count} entities, {len(sections)} sections")
        
        # Build comprehensive evaluation prompt
        prompt = self.prompt_template.format(
            QUESTION="Comprehensively evaluate the quality of knowledge graph extracted from academic papers across all dimensions",
            ONTOLOGY_OR_RULES="Standard knowledge graph evaluation criteria",
            SECTIONED_TURTLE_BLOCK=ttl_content
        )
        
        # Call model for evaluation
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional knowledge graph evaluation expert. Please return evaluation results in JSON format only as required."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            result_text = response.choices[0].message.content.strip()
            logger.info("GPT-4o evaluation completed")
            
            # Parse JSON result
            try:
                # Extract JSON part
                if result_text.startswith('```json'):
                    result_text = result_text[7:-3]
                elif result_text.startswith('```'):
                    result_text = result_text[3:-3]
                
                result_json = json.loads(result_text)
                
                # Create evaluation result object
                evaluation_result = EvaluationResult(
                    meta=result_json.get("meta", {}),
                    scores=result_json.get("scores", {}),
                    final_score=result_json.get("final_score", 0.0),
                    summary_advice=result_json.get("summary_advice", ""),
                    top_fixes=result_json.get("top_fixes", []),
                    passed_threshold=result_json.get("final_score", 0.0) >= self.threshold
                )
                
                return evaluation_result
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing GPT-4o response JSON: {e}")
                logger.error(f"Original response: {result_text}")
                raise
                
        except Exception as e:
            logger.error(f"Error calling GPT-4o API: {e}")
            raise
    
    def evaluate_and_save(self, ttl_file_path: str, output_path: Optional[str] = None) -> EvaluationResult:
        """
        Evaluate TTL file and save results
        
        Args:
            ttl_file_path: TTL file path
            output_path: Output file path, if None will auto-generate
            
        Returns:
            Evaluation result
        """
        # Perform comprehensive evaluation
        result = self.evaluate_ttl(ttl_file_path)
        
        # Prepare output results
        output_data = {
            "input_file": ttl_file_path,
            "evaluation_threshold": self.threshold,
            "evaluation_result": {
                "meta": result.meta,
                "scores": result.scores,
                "final_score": result.final_score,
                "summary_advice": result.summary_advice,
                "top_fixes": result.top_fixes,
                "passed_threshold": result.passed_threshold
            },
            "summary": {
                "score": result.final_score,
                "threshold": self.threshold,
                "passed": result.passed_threshold,
                "key_advice": result.summary_advice,
                "priority_fixes": result.top_fixes[:3]  # Keep only top 3 fix suggestions
            }
        }
        
        # Save results
        if output_path is None:
            ttl_file = Path(ttl_file_path)
            output_path = ttl_file.parent / f"{ttl_file.stem}_evaluation.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Evaluation results saved to: {output_path}")
        
        # Print summary results
        self._print_summary(result)
        
        return result
    
    def _print_summary(self, result: EvaluationResult):
        """Print evaluation result summary"""
        print("\n" + "="*60)
        print("TTL Knowledge Graph Evaluation Results")
        print("="*60)
        print(f" Final Score: {result.final_score:.1f}/10.0")
        print(f"Threshold Passed: {'Yes' if result.passed_threshold else 'No'} (threshold: {self.threshold})")
        print(f"\nOverall Advice: {result.summary_advice}")
        
        print(f"\nDimension Scores:")
        for dimension, score_info in result.scores.items():
            score = score_info.get('score', 0)
            reason = score_info.get('reason', '')
            print(f"  • {dimension}: {score:.1f}/10 ({reason})")
        
        print(f"\n Priority Fixes:")
        for i, fix in enumerate(result.top_fixes[:3], 1):
            print(f"  {i}. {fix}")
        
        print("="*60)


def main():
    """Main function example"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TTL File Knowledge Graph Evaluator")
    parser.add_argument("ttl_file", help="TTL file path")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("-t", "--threshold", type=float, default=0.8, help="Score threshold")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--base-url", help="OpenAI base URL")
    
    args = parser.parse_args()
    
    try:
        # Create evaluator
        evaluator = TTLEvaluator(api_key=args.api_key, base_url=args.base_url, threshold=args.threshold)
        
        # Perform comprehensive evaluation
        result = evaluator.evaluate_and_save(
            ttl_file_path=args.ttl_file,
            output_path=args.output
        )
        
        return 0 if result.passed_threshold else 1
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return 1


if __name__ == "__main__":
    exit(main())