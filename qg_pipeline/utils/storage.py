#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Storage and state management utilities for the multi-agent pipeline
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum


class ProcessingStatus(Enum):
    """Processing status enumeration"""
    PENDING = "pending"
    EXTRACTING = "extracting"
    EVALUATING = "evaluating"
    RETRYING = "retrying"
    GENERATING_QA = "generating_qa"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PaperState:
    """State information for a single paper"""
    paper_name: str
    paper_path: str
    status: ProcessingStatus
    current_step: str
    attempts: int
    max_attempts: int
    
    # File paths
    extraction_file: Optional[str] = None
    evaluation_file: Optional[str] = None
    qa_file: Optional[str] = None
    
    # Results
    extraction_successful: bool = False
    evaluation_passed: bool = False
    evaluation_score: Optional[float] = None
    qa_generated: bool = False
    
    # Metrics
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    total_duration: Optional[float] = None
    
    # Error information
    last_error: Optional[str] = None
    improvement_suggestions: Optional[str] = None
    
    # Statistics
    statistics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['status'] = self.status.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PaperState':
        """Create from dictionary"""
        data = data.copy()
        data['status'] = ProcessingStatus(data['status'])
        return cls(**data)


@dataclass
class BatchState:
    """State information for batch processing"""
    batch_id: str
    start_time: str
    end_time: Optional[str] = None
    total_papers: int = 0
    completed_papers: int = 0
    failed_papers: int = 0
    
    # Configuration
    config: Optional[Dict[str, Any]] = None
    
    # Paper states
    papers: Dict[str, PaperState] = None
    
    # Overall statistics
    overall_statistics: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.papers is None:
            self.papers = {}
    
    def add_paper(self, paper_state: PaperState):
        """Add paper state"""
        self.papers[paper_state.paper_name] = paper_state
        self.total_papers = len(self.papers)
    
    def update_paper(self, paper_name: str, **updates):
        """Update paper state"""
        if paper_name in self.papers:
            for key, value in updates.items():
                setattr(self.papers[paper_name], key, value)
    
    def get_paper(self, paper_name: str) -> Optional[PaperState]:
        """Get paper state"""
        return self.papers.get(paper_name)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get batch summary"""
        completed = sum(1 for p in self.papers.values() if p.status == ProcessingStatus.COMPLETED)
        failed = sum(1 for p in self.papers.values() if p.status == ProcessingStatus.FAILED)
        in_progress = self.total_papers - completed - failed
        
        return {
            'batch_id': self.batch_id,
            'total_papers': self.total_papers,
            'completed': completed,
            'failed': failed,
            'in_progress': in_progress,
            'success_rate': completed / self.total_papers if self.total_papers > 0 else 0,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'is_complete': completed + failed == self.total_papers
        }


class StateManager:
    """Manages workflow state persistence"""
    
    def __init__(self, state_dir: Path):
        """
        Initialize state manager
        
        Args:
            state_dir: Directory for state files
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        self.batch_state_file = self.state_dir / "batch_state.json"
        self.checkpoint_file = self.state_dir / "checkpoint.pickle"
    
    def create_batch_state(self, batch_id: str, config: Dict[str, Any]) -> BatchState:
        """
        Create new batch state
        
        Args:
            batch_id: Batch identifier
            config: Configuration dictionary
            
        Returns:
            New batch state
        """
        batch_state = BatchState(
            batch_id=batch_id,
            start_time=datetime.now().isoformat(),
            config=config
        )
        return batch_state
    
    def save_batch_state(self, batch_state: BatchState):
        """
        Save batch state to file
        
        Args:
            batch_state: Batch state to save
        """
        # Convert to serializable format
        state_data = {
            'batch_id': batch_state.batch_id,
            'start_time': batch_state.start_time,
            'end_time': batch_state.end_time,
            'total_papers': batch_state.total_papers,
            'completed_papers': batch_state.completed_papers,
            'failed_papers': batch_state.failed_papers,
            'config': batch_state.config,
            'papers': {name: paper.to_dict() for name, paper in batch_state.papers.items()},
            'overall_statistics': batch_state.overall_statistics
        }
        
        with open(self.batch_state_file, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, ensure_ascii=False, indent=2)
    
    def load_batch_state(self) -> Optional[BatchState]:
        """
        Load batch state from file
        
        Returns:
            Loaded batch state or None if not found
        """
        if not self.batch_state_file.exists():
            return None
        
        try:
            with open(self.batch_state_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            
            # Reconstruct batch state
            batch_state = BatchState(
                batch_id=state_data['batch_id'],
                start_time=state_data['start_time'],
                end_time=state_data.get('end_time'),
                total_papers=state_data.get('total_papers', 0),
                completed_papers=state_data.get('completed_papers', 0),
                failed_papers=state_data.get('failed_papers', 0),
                config=state_data.get('config'),
                overall_statistics=state_data.get('overall_statistics')
            )
            
            # Reconstruct paper states
            papers_data = state_data.get('papers', {})
            for name, paper_data in papers_data.items():
                batch_state.papers[name] = PaperState.from_dict(paper_data)
            
            return batch_state
            
        except Exception as e:
            print(f"Error loading batch state: {e}")
            return None
    
    def save_checkpoint(self, workflow_state: Dict[str, Any]):
        """
        Save workflow checkpoint
        
        Args:
            workflow_state: Current workflow state
        """
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(workflow_state, f)
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load workflow checkpoint
        
        Returns:
            Loaded checkpoint or None if not found
        """
        if not self.checkpoint_file.exists():
            return None
        
        try:
            with open(self.checkpoint_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None
    
    def cleanup_old_states(self, keep_days: int = 7):
        """
        Cleanup old state files
        
        Args:
            keep_days: Number of days to keep
        """
        cutoff_time = datetime.now().timestamp() - (keep_days * 24 * 3600)
        
        for file_path in self.state_dir.iterdir():
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                except Exception as e:
                    print(f"Error removing old state file {file_path}: {e}")


class ResultsManager:
    """Manages workflow results and statistics"""
    
    def __init__(self, results_dir: Path):
        """
        Initialize results manager
        
        Args:
            results_dir: Directory for results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def save_paper_results(self, paper_name: str, results: Dict[str, Any]):
        """
        Save results for a single paper
        
        Args:
            paper_name: Paper name
            results: Results dictionary
        """
        results_file = self.results_dir / f"{paper_name}_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    def generate_batch_report(self, batch_state: BatchState) -> Dict[str, Any]:
        """
        Generate comprehensive batch report
        
        Args:
            batch_state: Batch state
            
        Returns:
            Batch report dictionary
        """
        summary = batch_state.get_summary()
        
        # Detailed paper results
        paper_results = []
        for paper_name, paper_state in batch_state.papers.items():
            paper_results.append({
                'paper_name': paper_name,
                'status': paper_state.status.value,
                'attempts': paper_state.attempts,
                'extraction_successful': paper_state.extraction_successful,
                'evaluation_passed': paper_state.evaluation_passed,
                'evaluation_score': paper_state.evaluation_score,
                'qa_generated': paper_state.qa_generated,
                'duration': paper_state.total_duration,
                'error': paper_state.last_error,
                'statistics': paper_state.statistics
            })
        
        # Aggregate statistics
        total_extractions = sum(1 for p in batch_state.papers.values() if p.extraction_successful)
        total_evaluations_passed = sum(1 for p in batch_state.papers.values() if p.evaluation_passed)
        total_qa_generated = sum(1 for p in batch_state.papers.values() if p.qa_generated)
        
        avg_score = None
        scores = [p.evaluation_score for p in batch_state.papers.values() if p.evaluation_score is not None]
        if scores:
            avg_score = sum(scores) / len(scores)
        
        report = {
            'summary': summary,
            'statistics': {
                'total_papers': batch_state.total_papers,
                'successful_extractions': total_extractions,
                'passed_evaluations': total_evaluations_passed,
                'generated_qa': total_qa_generated,
                'extraction_success_rate': total_extractions / batch_state.total_papers if batch_state.total_papers > 0 else 0,
                'evaluation_pass_rate': total_evaluations_passed / batch_state.total_papers if batch_state.total_papers > 0 else 0,
                'qa_generation_rate': total_qa_generated / batch_state.total_papers if batch_state.total_papers > 0 else 0,
                'average_evaluation_score': avg_score
            },
            'paper_results': paper_results,
            'config': batch_state.config,
            'generated_at': datetime.now().isoformat()
        }
        
        return report
    
    def save_batch_report(self, batch_state: BatchState):
        """
        Save batch report to file
        
        Args:
            batch_state: Batch state
        """
        report = self.generate_batch_report(batch_state)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"batch_report_{batch_state.batch_id}_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        return report_file
