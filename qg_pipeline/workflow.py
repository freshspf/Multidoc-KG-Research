#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Agent Knowledge Graph Pipeline using LangGraph
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, TypedDict
from datetime import datetime
import traceback

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Local imports
from utils.helpers import (
    load_config, setup_logging, create_directories, 
    get_input_papers, generate_filename, Timer
)
from utils.storage import (
    StateManager, ResultsManager, BatchState, PaperState, 
    ProcessingStatus
)
from agents.extractor.section_based_extraction import SectionBasedExtractor
from agents.evaluator.ttl_evaluator import TTLEvaluator
from agents.QAgenerator.section_multi_hop_qa_generator import MultiHopQAGenerator


class WorkflowState(TypedDict):
    """State for the multi-agent workflow"""
    # Current paper being processed
    current_paper: Optional[str]
    current_paper_path: Optional[str]
    
    # Processing state
    batch_id: str
    paper_queue: List[str]
    completed_papers: List[str]
    failed_papers: List[str]
    
    # Current step results
    extraction_file: Optional[str]
    evaluation_result: Optional[Dict[str, Any]]
    qa_file: Optional[str]
    
    # Retry mechanism
    current_attempts: int
    max_attempts: int
    improvement_suggestions: Optional[str]
    
    # Error handling
    last_error: Optional[str]
    should_retry: bool
    
    # Statistics
    batch_statistics: Dict[str, Any]


class MultiAgentKGPipeline:
    """Multi-Agent Knowledge Graph Pipeline using LangGraph"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the multi-agent pipeline
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup directories
        self.directories = create_directories(self.config)
        
        # Setup logging
        log_file = self.directories['logs'] / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.logger = setup_logging(
            log_level=self.config.get('workflow', {}).get('log_level', 'INFO'),
            log_file=str(log_file)
        )
        
        # Initialize managers
        self.state_manager = StateManager(self.directories['output'] / 'state')
        self.results_manager = ResultsManager(self.directories['output'] / 'results')
        
        # Initialize agents
        self._initialize_agents()
        
        # Build workflow graph
        self.workflow = self._build_workflow()
        
        self.logger.info("Multi-Agent KG Pipeline initialized successfully")
    
    def _initialize_agents(self):
        """Initialize all agents"""
        try:
            # Initialize extractor with configuration
            self.extractor = SectionBasedExtractor(
                output_dir=str(self.directories['extraction']),
                config=self.config
            )
            self.logger.info(f"Extractor initialized with model: {self.extractor.model}")

            # Initialize evaluator with configuration
            self.evaluator = TTLEvaluator(
                threshold=self.config['evaluator']['threshold'],
                config=self.config
            )
            self.logger.info(f"Evaluator initialized with model: {self.evaluator.model}")

            # Initialize QA generator (lazy initialization due to API dependency)
            self.qa_generator = None
            self.logger.info("QA generator will be initialized when needed")

        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {e}")
            raise
    
    def _get_qa_generator(self):
        """Lazy initialization of QA generator"""
        if self.qa_generator is None:
            self.qa_generator = MultiHopQAGenerator(config=self.config)
            self.logger.info(f"QA generator initialized with model: {self.qa_generator.model}")
        return self.qa_generator
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        # Create workflow graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("initialize_batch", self.initialize_batch)
        workflow.add_node("select_next_paper", self.select_next_paper)
        workflow.add_node("extract_knowledge", self.extract_knowledge)
        workflow.add_node("evaluate_knowledge", self.evaluate_knowledge)
        workflow.add_node("decide_retry", self.decide_retry)
        workflow.add_node("generate_qa", self.generate_qa)
        workflow.add_node("finalize_paper", self.finalize_paper)
        workflow.add_node("finalize_batch", self.finalize_batch)
        
        # Set entry point
        workflow.set_entry_point("initialize_batch")
        
        # Add edges
        workflow.add_edge("initialize_batch", "select_next_paper")
        
        # Conditional routing from select_next_paper
        workflow.add_conditional_edges(
            "select_next_paper",
            self.route_after_selection,
            {
                "process_paper": "extract_knowledge",
                "batch_complete": "finalize_batch"
            }
        )
        
        workflow.add_edge("extract_knowledge", "evaluate_knowledge")
        
        # Conditional routing from evaluate_knowledge
        workflow.add_conditional_edges(
            "evaluate_knowledge",
            self.route_after_evaluation,
            {
                "retry": "decide_retry",
                "generate_qa": "generate_qa",
                "finalize": "finalize_paper"
            }
        )
        
        # Conditional routing from decide_retry
        workflow.add_conditional_edges(
            "decide_retry",
            self.route_after_retry_decision,
            {
                "retry_extraction": "extract_knowledge",
                "give_up": "finalize_paper"
            }
        )
        
        workflow.add_edge("generate_qa", "finalize_paper")
        
        # After finalizing a paper, go back to select next
        workflow.add_edge("finalize_paper", "select_next_paper")
        
        # End the workflow
        workflow.add_edge("finalize_batch", END)
        
        return workflow
    
    def initialize_batch(self, state: WorkflowState) -> WorkflowState:
        """Initialize batch processing"""
        self.logger.info("🚀 Initializing batch processing...")
        
        try:
            # Get input papers
            paper_files = get_input_papers(self.config)
            paper_queue = [str(p.name) for p in paper_files]
            
            # 详细记录论文队列
            self.logger.info(f"Paper Queue:")
            for i, paper in enumerate(paper_queue):
                self.logger.info(f"  {i+1}. {paper}")
            
            if not paper_queue:
                raise ValueError("No input papers found")
            
            # Create batch state
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            batch_state = self.state_manager.create_batch_state(batch_id, self.config)
            
            # Initialize paper states
            for paper_name in paper_queue:
                paper_path = str(Path(self.config['paths']['input_dir']) / paper_name)
                paper_state = PaperState(
                    paper_name=paper_name,
                    paper_path=paper_path,
                    status=ProcessingStatus.PENDING,
                    current_step="initialized",
                    attempts=0,
                    max_attempts=self.config['retry']['max_attempts'],
                    start_time=datetime.now().isoformat()
                )
                batch_state.add_paper(paper_state)
            
            # Save batch state
            self.state_manager.save_batch_state(batch_state)
            
            # Update workflow state
            state.update({
                "batch_id": batch_id,
                "paper_queue": paper_queue,
                "completed_papers": [],
                "failed_papers": [],
                "current_paper": None,
                "current_paper_path": None,
                "extraction_file": None,
                "evaluation_result": None,
                "qa_file": None,
                "current_attempts": 0,
                "max_attempts": self.config['retry']['max_attempts'],
                "improvement_suggestions": None,
                "last_error": None,
                "should_retry": False,
                "batch_statistics": {
                    "start_time": datetime.now().isoformat(),
                    "total_papers": len(paper_queue),
                    "processed": 0
                }
            })
            
            self.logger.info(f"Batch initialized: {len(paper_queue)} papers to process")
            return state
            
        except Exception as e:
            self.logger.error(f"Failed to initialize batch: {e}")
            state["last_error"] = str(e)
            return state
    
    def select_next_paper(self, state: WorkflowState) -> WorkflowState:
        """Select next paper to process"""
        if not state["paper_queue"]:
            self.logger.info("📋 No more papers to process")
            # 清除当前论文状态，确保路由到batch_complete
            state.update({
                "current_paper": None,
                "current_paper_path": None,
                "extraction_file": None,
                "evaluation_result": None,
                "qa_file": None,
                "current_attempts": 0,
                "improvement_suggestions": None,
                "last_error": None,
                "should_retry": False
            })
            return state
        
        # Get next paper
        next_paper = state["paper_queue"].pop(0)
        paper_path = str(Path(self.config['paths']['input_dir']) / next_paper)
        
        # 检查文件是否存在
        if not Path(paper_path).exists():
            self.logger.error(f"File not found: {paper_path}")
            # 将论文标记为失败并继续下一个
            state["failed_papers"].append(next_paper)
            state["last_error"] = f"File not found: {paper_path}"
            return state
        
        # 记录当前论文队列状态（调试用）
        self.logger.info(f"Current paper queue state:")
        for i, paper in enumerate(state["paper_queue"][:5]):  # 只显示前5个
            self.logger.info(f"  Queue[{i}]: {paper}")
        if len(state["paper_queue"]) > 5:
            self.logger.info(f"  ... {len(state['paper_queue']) - 5} more papers")
        
        # 详细记录论文选择过程
        remaining_papers = len(state["paper_queue"])
        self.logger.info(f"Selected paper: {next_paper}")
        self.logger.info(f"Remaining papers: {remaining_papers}")
        if remaining_papers > 0:
            self.logger.info(f"Next paper will be: {state['paper_queue'][0]}")
        
        # Update state
        state.update({
            "current_paper": next_paper,
            "current_paper_path": paper_path,
            "current_attempts": 0,
            "extraction_file": None,
            "evaluation_result": None,
            "qa_file": None,
            "improvement_suggestions": None,
            "last_error": None,
            "should_retry": False
        })
        
        # Update paper state
        batch_state = self.state_manager.load_batch_state()
        if batch_state:
            batch_state.update_paper(
                next_paper,
                status=ProcessingStatus.EXTRACTING,
                current_step="selected",
                start_time=datetime.now().isoformat()
            )
            self.state_manager.save_batch_state(batch_state)
        return state
    
    def route_after_selection(self, state: WorkflowState) -> str:
        """Route after paper selection"""
        if state.get("current_paper"):
            return "process_paper"
        else:
            return "batch_complete"
    
    def extract_knowledge(self, state: WorkflowState) -> WorkflowState:
        """Extract knowledge graph from paper"""
        paper_name = state["current_paper"]
        paper_path = state["current_paper_path"]
        
        self.logger.info(f"Extracting knowledge from: {paper_name}")
        
        try:
            with Timer(f"Knowledge extraction for {paper_name}") as timer:
                # Prepare evaluation result for improvement suggestions
                evaluation_result = None
                if state.get("improvement_suggestions"):
                    evaluation_result = {
                        "passed": False,
                        "suggestions": state["improvement_suggestions"]
                    }
                
                # Extract knowledge
                result = self.extractor.process_single_paper(paper_path, evaluation_result)
                
                if result.get("success"):
                    state["extraction_file"] = result["saved_file"]
                    
                    # Update paper state
                    batch_state = self.state_manager.load_batch_state()
                    if batch_state:
                        batch_state.update_paper(
                            paper_name,
                            status=ProcessingStatus.EVALUATING,
                            current_step="extracted",
                            extraction_file=result["saved_file"],
                            extraction_successful=True,
                            statistics=result.get("statistics")
                        )
                        self.state_manager.save_batch_state(batch_state)
                    
                    self.logger.info(f" Knowledge extraction completed: {timer.get_duration_str()}")
                else:
                    raise Exception(result.get("error", "Unknown extraction error"))
        
        except Exception as e:
            error_msg = f"Knowledge extraction failed: {e}"
            self.logger.error(f"error: {error_msg}")
            state["last_error"] = error_msg
            
            # Update paper state
            batch_state = self.state_manager.load_batch_state()
            if batch_state:
                batch_state.update_paper(
                    paper_name,
                    status=ProcessingStatus.FAILED,
                    current_step="extraction_failed",
                    last_error=error_msg,
                    extraction_successful=False
                )
                self.state_manager.save_batch_state(batch_state)
        
        return state
    
    def evaluate_knowledge(self, state: WorkflowState) -> WorkflowState:
        """Evaluate extracted knowledge graph"""
        paper_name = state["current_paper"]
        extraction_file = state.get("extraction_file")
        
        if not extraction_file:
            state["last_error"] = "No extraction file available for evaluation"
            return state
        
        self.logger.info(f"Evaluating knowledge graph: {paper_name}")
        
        try:
            with Timer(f"Knowledge evaluation for {paper_name}") as timer:
                # Evaluate the TTL file
                result = self.evaluator.evaluate_ttl(extraction_file)
                
                # Create evaluation result
                evaluation_result = {
                    "passed": result.passed_threshold,
                    "score": result.final_score,
                    "threshold": self.evaluator.threshold,
                    "summary_advice": result.summary_advice,
                    "top_fixes": result.top_fixes,
                    "meta": result.meta,
                    "scores": result.scores
                }

                state["evaluation_result"] = evaluation_result

                # Save evaluation result to evaluations directory
                self._save_evaluation_result(paper_name, evaluation_result)
                
                # Prepare improvement suggestions for retry
                if not result.passed_threshold:
                    suggestions = result.summary_advice
                    if result.top_fixes:
                        suggestions += " Priority fixes: " + "; ".join(result.top_fixes[:3])
                    state["improvement_suggestions"] = suggestions
                
                # Update paper state
                batch_state = self.state_manager.load_batch_state()
                if batch_state:
                    batch_state.update_paper(
                        paper_name,
                        current_step="evaluated",
                        evaluation_passed=result.passed_threshold,
                        evaluation_score=result.final_score,
                        improvement_suggestions=state.get("improvement_suggestions")
                    )
                    self.state_manager.save_batch_state(batch_state)
                
                if result.passed_threshold:
                    self.logger.info(f"Evaluation passed: {result.final_score:.1f}/10.0 ({timer.get_duration_str()})")
                else:
                    self.logger.warning(f"Evaluation failed: {result.final_score:.1f}/10.0 ({timer.get_duration_str()})")
        
        except Exception as e:
            error_msg = f"Knowledge evaluation failed: {e}"
            self.logger.error(f"error: {error_msg}")
            state["last_error"] = error_msg
            state["evaluation_result"] = {"passed": False, "error": error_msg}
        
        return state
    
    def route_after_evaluation(self, state: WorkflowState) -> str:
        """Route after evaluation"""
        evaluation_result = state.get("evaluation_result", {})
        
        # Handle case where evaluation_result might be None
        if evaluation_result and evaluation_result.get("passed"):
            return "generate_qa"
        elif state["current_attempts"] < state["max_attempts"] and self.config['retry']['enable_improvement_suggestions']:
            return "retry"
        else:
            return "finalize"
    
    def decide_retry(self, state: WorkflowState) -> WorkflowState:
        """Decide whether to retry extraction"""
        state["current_attempts"] += 1
        
        # Update paper state
        paper_name = state["current_paper"]
        batch_state = self.state_manager.load_batch_state()
        if batch_state:
            batch_state.update_paper(
                paper_name,
                status=ProcessingStatus.RETRYING,
                current_step="deciding_retry",
                attempts=state["current_attempts"]
            )
            self.state_manager.save_batch_state(batch_state)
        
        if state["current_attempts"] < state["max_attempts"]:
            state["should_retry"] = True
            self.logger.info(f"Retrying extraction for {paper_name} (attempt {state['current_attempts']}/{state['max_attempts']})")
        else:
            state["should_retry"] = False
            self.logger.warning(f"Max retry attempts reached for {paper_name}")
        
        return state
    
    def route_after_retry_decision(self, state: WorkflowState) -> str:
        """Route after retry decision"""
        if state.get("should_retry", False):
            return "retry_extraction"
        else:
            return "give_up"
    
    def generate_qa(self, state: WorkflowState) -> WorkflowState:
        """Generate QA pairs from knowledge graph"""
        paper_name = state["current_paper"]
        extraction_file = state.get("extraction_file")
        
        if not extraction_file:
            state["last_error"] = "No extraction file available for QA generation"
            return state
        
        self.logger.info(f" Generating QA pairs: {paper_name}")
        
        try:
            with Timer(f"QA generation for {paper_name}") as timer:
                # Get QA generator
                qa_gen = self._get_qa_generator()
                
                # Generate QA pairs
                qa_gen.run_pipeline(
                    ttl_file_path=extraction_file,
                    max_paths_per_section=self.config['qa_generator']['max_paths_per_section'],
                    max_qa_per_section=self.config['qa_generator']['max_qa_per_section']
                )
                
                # Extract paper base name for QA file
                from pathlib import Path
                paper_base_name = Path(paper_name).stem
                
                # QA file path (the generator saves with this naming pattern)
                qa_file = str(self.directories['qa'] / f"{paper_base_name}_qa_simplified.json")
                state["qa_file"] = qa_file
                
                # Update paper state
                batch_state = self.state_manager.load_batch_state()
                if batch_state:
                    batch_state.update_paper(
                        paper_name,
                        status=ProcessingStatus.GENERATING_QA,
                        current_step="qa_generated",
                        qa_file=qa_file,
                        qa_generated=True
                    )
                    self.state_manager.save_batch_state(batch_state)
                
                self.logger.info(f" QA generation completed: {timer.get_duration_str()}")
        
        except Exception as e:
            error_msg = f"QA generation failed: {e}"
            self.logger.error(f"error: {error_msg}")
            state["last_error"] = error_msg
            
            # Even if QA generation fails, we don't consider the paper as failed
            # since extraction and evaluation were successful
        
        return state
    
    def finalize_paper(self, state: WorkflowState) -> WorkflowState:
        """Finalize processing for current paper"""
        paper_name = state["current_paper"]
        
        # 详细记录论文完成过程
        self.logger.info(f"Completed paper processing: {paper_name}")
        
        # Determine final status
        evaluation_result = state.get("evaluation_result", {})
        qa_file = state.get("qa_file")
        last_error = state.get("last_error")
        
        if last_error and not evaluation_result.get("passed"):
            # Paper failed
            final_status = ProcessingStatus.FAILED
            state["failed_papers"].append(paper_name)
            self.logger.error(f"Paper failed: {paper_name} - error: {last_error}")
        else:
            # Paper completed (successfully extracted and evaluated, QA optional)
            final_status = ProcessingStatus.COMPLETED
            state["completed_papers"].append(paper_name)
            self.logger.info(f"Paper completed: {paper_name}")
        
        # 记录当前状态统计
        completed_count = len(state["completed_papers"])
        failed_count = len(state["failed_papers"])
        remaining_count = len(state["paper_queue"])
        self.logger.info(f"Current statistics - completed: {completed_count}, failed: {failed_count}, remaining: {remaining_count}")
        
        # Update paper state
        batch_state = self.state_manager.load_batch_state()
        if batch_state:
            batch_state.update_paper(
                paper_name,
                status=final_status,
                current_step="finalized",
                end_time=datetime.now().isoformat()
            )
            
            # Calculate duration
            paper_state = batch_state.get_paper(paper_name)
            if paper_state and paper_state.start_time:
                start_dt = datetime.fromisoformat(paper_state.start_time)
                end_dt = datetime.fromisoformat(paper_state.end_time)
                duration = (end_dt - start_dt).total_seconds()
                batch_state.update_paper(paper_name, total_duration=duration)
            
            self.state_manager.save_batch_state(batch_state)
        
        # Update batch statistics
        state["batch_statistics"]["processed"] += 1
        
        # Reset current paper state
        state.update({
            "current_paper": None,
            "current_paper_path": None,
            "extraction_file": None,
            "evaluation_result": None,
            "qa_file": None,
            "current_attempts": 0,
            "improvement_suggestions": None,
            "last_error": None,
            "should_retry": False
        })
        
        return state
    
    def finalize_batch(self, state: WorkflowState) -> WorkflowState:
        """Finalize batch processing"""
        self.logger.info("Finalizing batch processing...")
        
        try:
            # Update batch statistics
            state["batch_statistics"]["end_time"] = datetime.now().isoformat()
            state["batch_statistics"]["completed"] = len(state["completed_papers"])
            state["batch_statistics"]["failed"] = len(state["failed_papers"])
            
            # Update and save final batch state
            batch_state = self.state_manager.load_batch_state()
            if batch_state:
                batch_state.end_time = datetime.now().isoformat()
                batch_state.completed_papers = len(state["completed_papers"])
                batch_state.failed_papers = len(state["failed_papers"])
                batch_state.overall_statistics = state["batch_statistics"]
                self.state_manager.save_batch_state(batch_state)
                
                # Generate and save batch report
                report_file = self.results_manager.save_batch_report(batch_state)
                self.logger.info(f"Batch report saved: {report_file}")
            
            # Print summary
            total = state["batch_statistics"]["total_papers"]
            completed = len(state["completed_papers"])
            failed = len(state["failed_papers"])
            success_rate = (completed / total * 100) if total > 0 else 0
            
            self.logger.info("="*60)
            self.logger.info("BATCH PROCESSING SUMMARY")
            self.logger.info("="*60)
            self.logger.info(f"Total papers: {total}")
            self.logger.info(f"Completed: {completed}")
            self.logger.info(f"Failed: {failed}")
            self.logger.info(f"Success rate: {success_rate:.1f}%")
            self.logger.info(f" Start time: {state['batch_statistics']['start_time']}")
            self.logger.info(f"End time: {state['batch_statistics']['end_time']}")
            self.logger.info("="*60)
            
        except Exception as e:
            self.logger.error(f"Failed to finalize batch: {e}")
            state["last_error"] = str(e)
        
        return state
    
    def run_batch(self, resume_from_checkpoint: bool = False) -> Dict[str, Any]:
        """
        Run batch processing
        
        Args:
            resume_from_checkpoint: Whether to resume from checkpoint
            
        Returns:
            Batch processing results
        """
        try:
            # Initialize workflow with memory
            memory = MemorySaver()
            app = self.workflow.compile(checkpointer=memory)
            
            # Create initial state
            initial_state = WorkflowState(
                current_paper=None,
                current_paper_path=None,
                batch_id="",
                paper_queue=[],
                completed_papers=[],
                failed_papers=[],
                extraction_file=None,
                evaluation_result=None,
                qa_file=None,
                current_attempts=0,
                max_attempts=0,
                improvement_suggestions=None,
                last_error=None,
                should_retry=False,
                batch_statistics={}
            )
            
            # Run workflow with proper recursion limit
            thread_config = {
                "configurable": {
                    "thread_id": f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                }
            }
            
            final_state = None
            iteration_count = 0
            # 根据LangGraph官方文档，正确设置递归限制
            config = {"recursion_limit": 100}
            config.update(thread_config)
            
            for state in app.stream(initial_state, config):
                final_state = state
                iteration_count += 1
                
                                # 添加详细的调试信息
                if self.config.get('workflow', {}).get('verbose_logging', True):
                    # 获取实际的状态值（从不同的键中）
                    actual_state = {}
                    for key, value in state.items():
                        if isinstance(value, dict):
                            actual_state.update(value)
                    
                    current_step = list(state.keys())[0] if state else "unknown"
                    paper_queue_len = len(actual_state.get("paper_queue", []))
                    completed_len = len(actual_state.get("completed_papers", []))
                    failed_len = len(actual_state.get("failed_papers", []))
                    
                    self.logger.info(f"Iteration {iteration_count}: {current_step} | "
                                   f"Queue: {paper_queue_len} | Completed: {completed_len} | Failed: {failed_len}")
                    
                    # 调试：显示当前状态的所有键
                    if iteration_count == 1:
                        self.logger.info(f"State keys: {list(state.keys())}")
                        self.logger.info(f"Actual state keys: {list(actual_state.keys())}")
                        self.logger.info(f"Paper queue content: {actual_state.get('paper_queue', [])}")
                    
                    # 如果迭代次数过多，记录警告
                    if iteration_count > 50:
                        self.logger.warning(f"Iteration count too high: {iteration_count}, possibly infinite loop")
                    
                    # LangGraph会自然运行到END状态，不需要提前退出
                    # 只在明显异常的情况下才强制退出
            
            return {
                "success": True,
                "final_state": final_state,
                "batch_statistics": final_state.get("batch_statistics", {}) if final_state else {}
            }
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            self.logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    def _save_evaluation_result(self, paper_name: str, evaluation_result: Dict[str, Any]):
        """Save evaluation result to evaluations directory"""
        import json

        # Create evaluation filename
        paper_base_name = paper_name.replace('.json', '')
        evaluation_file = self.directories['evaluation'] / f"{paper_base_name}_evaluation.json"

        # Prepare output data
        output_data = {
            "paper_name": paper_name,
            "evaluation_timestamp": datetime.now().isoformat(),
            "evaluation_result": evaluation_result,
            "summary": {
                "score": evaluation_result["score"],
                "threshold": evaluation_result["threshold"],
                "passed": evaluation_result["passed"],
                "key_advice": evaluation_result["summary_advice"],
                "priority_fixes": evaluation_result["top_fixes"][:3] if evaluation_result["top_fixes"] else []
            }
        }

        # Save to file
        with open(evaluation_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Evaluation result saved to: {evaluation_file}")


def main():
    """Main function to run the pipeline"""
    try:
        # Initialize pipeline
        print("Starting Multi-Agent Knowledge Graph Pipeline...")
        pipeline = MultiAgentKGPipeline()
        
        # Run batch processing
        result = pipeline.run_batch()
        
        if result["success"]:
            print("Pipeline completed successfully!")
            return 0
        else:
            print(f"Pipeline failed: {result['error']}")
            return 1
            
    except Exception as e:
        print(f"Pipeline initialization failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
