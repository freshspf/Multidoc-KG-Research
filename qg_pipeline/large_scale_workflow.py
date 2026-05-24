#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Large Scale Multi-Agent Knowledge Graph Pipeline
支持处理1000+论文的优化版本
"""

import os
import sys
import math
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, TypedDict, Iterator
from datetime import datetime
import traceback
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue

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


class LargeScaleWorkflowState(TypedDict):
    """大规模处理的工作流状态"""
    # 批次信息
    batch_id: str
    sub_batch_id: str
    sub_batch_index: int
    total_sub_batches: int
    
    # 当前处理状态
    current_paper: Optional[str]
    current_paper_path: Optional[str]
    
    # 队列管理（小批次）
    current_batch_papers: List[str]
    completed_papers: List[str]
    failed_papers: List[str]
    
    # 处理结果
    extraction_file: Optional[str]
    evaluation_result: Optional[Dict[str, Any]]
    qa_file: Optional[str]
    
    # 重试机制
    current_attempts: int
    max_attempts: int
    improvement_suggestions: Optional[str]
    
    # 错误处理
    last_error: Optional[str]
    should_retry: bool
    
    # 统计信息
    batch_statistics: Dict[str, Any]


class LargeScaleKGPipeline:
    """大规模知识图谱流水线"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """初始化大规模流水线"""
        # 加载配置
        self.config = load_config(config_path)
        
        # 大规模处理配置
        self.batch_size = self.config.get('large_scale', {}).get('batch_size', 50)
        self.max_concurrent = self.config.get('large_scale', {}).get('max_concurrent', 4)
        self.checkpoint_interval = self.config.get('large_scale', {}).get('checkpoint_interval', 10)
        
        # 设置目录
        self.directories = create_directories(self.config)
        
        # 设置日志
        log_file = self.directories['logs'] / f"large_scale_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.logger = setup_logging(
            log_level=self.config.get('workflow', {}).get('log_level', 'INFO'),
            log_file=str(log_file)
        )
        
        # 初始化管理器
        self.state_manager = StateManager(self.directories['output'] / 'state')
        self.results_manager = ResultsManager(self.directories['output'] / 'results')
        
        # 初始化代理
        self._initialize_agents()
        
        # 构建工作流
        self.workflow = self._build_workflow()
        
        self.logger.info("Initialize large-scale knowledge graph pipeline successfully")
    
    def _initialize_agents(self):
        """初始化所有代理"""
        try:
            # 初始化提取器
            self.extractor = SectionBasedExtractor(
                output_dir=str(self.directories['extraction']),
                config=self.config
            )
            self.logger.info(f"Extractor initialized with model: {self.extractor.model}")

            # 初始化评估器
            self.evaluator = TTLEvaluator(
                threshold=self.config['evaluator']['threshold'],
                config=self.config
            )
            self.logger.info(f"Evaluator initialized with model: {self.evaluator.model}")

            # QA生成器（延迟初始化）
            self.qa_generator = None
            self.logger.info("QA generator will be initialized when needed")

        except Exception as e:
            self.logger.error(f"Agent initialization failed: {e}")
            raise
    
    def _get_qa_generator(self):
        """延迟初始化QA生成器"""
        if self.qa_generator is None:
            self.qa_generator = MultiHopQAGenerator(config=self.config)
            self.logger.info(f"QA generator initialized with model: {self.qa_generator.model}")
        return self.qa_generator
    
    def _build_workflow(self) -> StateGraph:
        """构建LangGraph工作流"""
        # 创建工作流图
        workflow = StateGraph(LargeScaleWorkflowState)
        
        # 添加节点
        workflow.add_node("initialize_sub_batch", self.initialize_sub_batch)
        workflow.add_node("select_next_paper", self.select_next_paper)
        workflow.add_node("extract_knowledge", self.extract_knowledge)
        workflow.add_node("evaluate_knowledge", self.evaluate_knowledge)
        workflow.add_node("decide_retry", self.decide_retry)
        workflow.add_node("generate_qa", self.generate_qa)
        workflow.add_node("finalize_paper", self.finalize_paper)
        workflow.add_node("finalize_sub_batch", self.finalize_sub_batch)
        
        # 设置入口点
        workflow.set_entry_point("initialize_sub_batch")
        
        # 添加边
        workflow.add_edge("initialize_sub_batch", "select_next_paper")
        
        # 条件路由
        workflow.add_conditional_edges(
            "select_next_paper",
            self.route_after_selection,
            {
                "process_paper": "extract_knowledge",
                "sub_batch_complete": "finalize_sub_batch"
            }
        )
        
        workflow.add_edge("extract_knowledge", "evaluate_knowledge")
        
        workflow.add_conditional_edges(
            "evaluate_knowledge",
            self.route_after_evaluation,
            {
                "retry": "decide_retry",
                "generate_qa": "generate_qa",
                "finalize": "finalize_paper"
            }
        )
        
        workflow.add_conditional_edges(
            "decide_retry",
            self.route_after_retry_decision,
            {
                "retry_extraction": "extract_knowledge",
                "give_up": "finalize_paper"
            }
        )
        
        workflow.add_edge("generate_qa", "finalize_paper")
        workflow.add_edge("finalize_paper", "select_next_paper")
        workflow.add_edge("finalize_sub_batch", END)
        
        return workflow
    
    def split_papers_into_batches(self, papers: List[Path]) -> List[List[str]]:
        """将论文分割为小批次"""
        paper_names = [str(p.name) for p in papers]
        
        # 计算批次数量
        num_batches = math.ceil(len(paper_names) / self.batch_size)
        
        batches = []
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(paper_names))
            batch = paper_names[start_idx:end_idx]
            batches.append(batch)
        
        self.logger.info(f"Paper batching strategy: {len(paper_names)} papers → {num_batches} batches (each batch ≤ {self.batch_size} papers)")
        
        return batches
    
    def process_single_batch(self, batch_papers: List[str], batch_index: int, total_batches: int) -> Dict[str, Any]:
        """处理单个批次"""
        self.logger.info(f"Starting batch {batch_index + 1}/{total_batches} ({len(batch_papers)} papers)")
        
        try:
            # 初始化工作流
            memory = MemorySaver()
            app = self.workflow.compile(checkpointer=memory)
            
            # 创建初始状态
            batch_id = f"large_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            sub_batch_id = f"{batch_id}_sub_{batch_index + 1}"
            
            initial_state = LargeScaleWorkflowState(
                batch_id=batch_id,
                sub_batch_id=sub_batch_id,
                sub_batch_index=batch_index,
                total_sub_batches=total_batches,
                current_paper=None,
                current_paper_path=None,
                current_batch_papers=batch_papers.copy(),
                completed_papers=[],
                failed_papers=[],
                extraction_file=None,
                evaluation_result=None,
                qa_file=None,
                current_attempts=0,
                max_attempts=self.config['retry']['max_attempts'],
                improvement_suggestions=None,
                last_error=None,
                should_retry=False,
                batch_statistics={
                    "start_time": datetime.now().isoformat(),
                    "total_papers": len(batch_papers),
                    "processed": 0
                }
            )
            
            # 运行工作流
            thread_config = {
                "configurable": {
                    "thread_id": sub_batch_id
                }
            }
            
            # 设置递归限制（每个批次最多需要的步骤数）
            max_steps_per_paper = 15  # 每篇论文最多15步（包括重试）
            recursion_limit = len(batch_papers) * max_steps_per_paper + 10
            config = {"recursion_limit": recursion_limit}
            config.update(thread_config)
            
            self.logger.info(f"Batch configuration: recursion limit={recursion_limit}, number of papers={len(batch_papers)}")
            
            final_state = None
            for state in app.stream(initial_state, config):
                final_state = state
                
                # 提取实际状态
                actual_state = {}
                for key, value in state.items():
                    if isinstance(value, dict):
                        actual_state.update(value)
                
                # 进度监控
                completed = len(actual_state.get("completed_papers", []))
                failed = len(actual_state.get("failed_papers", []))
                remaining = len(actual_state.get("current_batch_papers", []))
                
                if completed + failed > 0 and (completed + failed) % self.checkpoint_interval == 0:
                    self.logger.info(f"Batch progress: completed={completed}, failed={failed}, remaining={remaining}")
            
            # 处理结果
            if final_state:
                actual_final_state = {}
                for key, value in final_state.items():
                    if isinstance(value, dict):
                        actual_final_state.update(value)
                
                return {
                    "success": True,
                    "batch_index": batch_index,
                    "completed": len(actual_final_state.get("completed_papers", [])),
                    "failed": len(actual_final_state.get("failed_papers", [])),
                    "completed_papers": actual_final_state.get("completed_papers", []),
                    "failed_papers": actual_final_state.get("failed_papers", []),
                    "statistics": actual_final_state.get("batch_statistics", {})
                }
            else:
                return {"success": False, "error": "Workflow did not return final state"}
                
        except Exception as e:
            self.logger.error(f"Batch {batch_index + 1} processing failed: {e}")
            self.logger.error(traceback.format_exc())
            return {
                "success": False,
                "batch_index": batch_index,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def run_large_scale_batch(self, parallel: bool = True) -> Dict[str, Any]:
        """运行大规模批处理"""
        start_time = datetime.now()
        self.logger.info("Starting large-scale batch processing...")
        
        try:
            # 获取所有论文
            papers = get_input_papers(self.config)
            if not papers:
                raise ValueError("No input papers found")
            
            self.logger.info(f"Found {len(papers)} papers to process")
            
            # 分割为批次
            paper_batches = self.split_papers_into_batches(papers)
            
            # 初始化结果
            all_results = []
            total_completed = 0
            total_failed = 0
            all_completed_papers = []
            all_failed_papers = []
            
            if parallel and len(paper_batches) > 1:
                # 并行处理多个批次
                self.logger.info(f"Parallel processing mode: maximum concurrent count={self.max_concurrent}")
                
                with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
                    # 提交所有批次任务
                    future_to_batch = {
                        executor.submit(self.process_single_batch, batch, i, len(paper_batches)): i
                        for i, batch in enumerate(paper_batches)
                    }
                    
                    # 收集结果
                    for future in as_completed(future_to_batch):
                        batch_index = future_to_batch[future]
                        try:
                            result = future.result()
                            all_results.append(result)
                            
                            if result["success"]:
                                total_completed += result["completed"]
                                total_failed += result["failed"]
                                all_completed_papers.extend(result["completed_papers"])
                                all_failed_papers.extend(result["failed_papers"])
                                self.logger.info(f"Batch {batch_index + 1} completed: {result['completed']} success, {result['failed']} failed")
                            else:
                                self.logger.error(f"Batch {batch_index + 1} failed: {result['error']}")
                                
                        except Exception as e:
                            self.logger.error(f"Batch {batch_index + 1} exception: {e}")
                            all_results.append({"success": False, "batch_index": batch_index, "error": str(e)})
            
            else:
                # 串行处理
                self.logger.info("Serial processing mode")
                
                for i, batch in enumerate(paper_batches):
                    result = self.process_single_batch(batch, i, len(paper_batches))
                    all_results.append(result)
                    
                    if result["success"]:
                        total_completed += result["completed"]
                        total_failed += result["failed"]
                        all_completed_papers.extend(result["completed_papers"])
                        all_failed_papers.extend(result["failed_papers"])
                        self.logger.info(f"Batch {i + 1} completed: {result['completed']} success, {result['failed']} failed")
                    else:
                        self.logger.error(f"Batch {i + 1} failed: {result['error']}")
            
            # 计算总体统计
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            success_rate = (total_completed / len(papers) * 100) if len(papers) > 0 else 0
            
            # 生成最终报告
            final_report = {
                "success": True,
                "total_papers": len(papers),
                "total_completed": total_completed,
                "total_failed": total_failed,
                "success_rate": success_rate,
                "total_batches": len(paper_batches),
                "batch_size": self.batch_size,
                "parallel_processing": parallel,
                "max_concurrent": self.max_concurrent if parallel else 1,
                "duration_seconds": duration,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "completed_papers": all_completed_papers,
                "failed_papers": all_failed_papers,
                "batch_results": all_results
            }
            
            # 保存报告
            report_file = self.directories['output'] / f"large_scale_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(final_report, f, indent=2, ensure_ascii=False)
            
            # 打印总结
            self.logger.info("="*80)
            self.logger.info("Large-scale batch processing summary")
            self.logger.info("="*80)
            self.logger.info(f"Total papers: {len(papers)}")
            self.logger.info(f"Processing completed: {total_completed}")
            self.logger.info(f"Processing failed: {total_failed}")
            self.logger.info(f"Success rate: {success_rate:.1f}%")
            self.logger.info(f"Total batches: {len(paper_batches)}")
            self.logger.info(f"Processing time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            self.logger.info(f"Average per paper: {duration/len(papers):.1f} seconds")
            self.logger.info(f"Report saved to: {report_file}")
            self.logger.info("="*80)
            
            return final_report
            
        except Exception as e:
            self.logger.error(f"Large-scale batch processing failed: {e}")
            self.logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    # ==================== 工作流节点方法 ====================
    
    def initialize_sub_batch(self, state: LargeScaleWorkflowState) -> LargeScaleWorkflowState:
        """初始化子批次"""
        sub_batch_id = state["sub_batch_id"]
        papers = state["current_batch_papers"]
        
        self.logger.info(f"Initializing sub-batch: {sub_batch_id}")
        self.logger.info(f"Sub-batch papers ({len(papers)} papers):")
        for i, paper in enumerate(papers):
            self.logger.info(f"  {i+1}. {paper}")
        
        state["batch_statistics"]["start_time"] = datetime.now().isoformat()
        return state
    
    def select_next_paper(self, state: LargeScaleWorkflowState) -> LargeScaleWorkflowState:
        """选择下一篇论文"""
        if not state["current_batch_papers"]:
            self.logger.info("No more papers to process in sub-batch")
            return state
        
        # 获取下一篇论文
        next_paper = state["current_batch_papers"].pop(0)
        paper_path = str(Path(self.config['paths']['input_dir']) / next_paper)
        
        # 检查文件是否存在
        if not Path(paper_path).exists():
            self.logger.error(f"Paper file not found: {paper_path}")
            state["failed_papers"].append(next_paper)
            state["last_error"] = f"File not found: {paper_path}"
            return state
        
        # 更新状态
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
        
        remaining = len(state["current_batch_papers"])
        self.logger.info(f"Selected paper: {next_paper} (remaining: {remaining})")
        
        return state
    
    def route_after_selection(self, state: LargeScaleWorkflowState) -> str:
        """选择后的路由"""
        if state.get("current_paper"):
            return "process_paper"
        else:
            return "sub_batch_complete"
    
    def extract_knowledge(self, state: LargeScaleWorkflowState) -> LargeScaleWorkflowState:
        """提取知识图谱"""
        paper_name = state["current_paper"]
        paper_path = state["current_paper_path"]
        
        self.logger.info(f"Extracting knowledge: {paper_name}")
        
        try:
            with Timer(f"Knowledge extraction {paper_name}") as timer:
                # 准备评估结果用于改进建议
                evaluation_result = None
                if state.get("improvement_suggestions"):
                    evaluation_result = {
                        "passed": False,
                        "suggestions": state["improvement_suggestions"]
                    }
                
                # 提取知识
                result = self.extractor.process_single_paper(paper_path, evaluation_result)
                
                if result.get("success"):
                    state["extraction_file"] = result["saved_file"]
                    self.logger.info(f"Knowledge extraction completed: {paper_name} ({timer.get_duration_str()})")
                else:
                    raise Exception(result.get("error", "Unknown extraction error"))
        
        except Exception as e:
            error_msg = f"Knowledge extraction failed: {e}"
            self.logger.error(f"{error_msg}")
            state["last_error"] = error_msg
        
        return state
    
    def evaluate_knowledge(self, state: LargeScaleWorkflowState) -> LargeScaleWorkflowState:
        """评估知识图谱"""
        paper_name = state["current_paper"]
        extraction_file = state.get("extraction_file")
        
        if not extraction_file:
            state["last_error"] = "No extraction file available for evaluation"
            return state
        
        self.logger.info(f"Evaluating knowledge graph: {paper_name}")
        
        try:
            with Timer(f"Knowledge evaluation {paper_name}") as timer:
                # 评估TTL文件
                result = self.evaluator.evaluate_ttl(extraction_file)
                
                # 创建评估结果
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
                
                # 准备改进建议用于重试
                if not result.passed_threshold:
                    suggestions = result.summary_advice
                    if result.top_fixes:
                        suggestions += " Prioritize fixing: " + "; ".join(result.top_fixes[:3])
                    state["improvement_suggestions"] = suggestions
                
                if result.passed_threshold:
                    self.logger.info(f"Evaluation passed: {paper_name} ({result.final_score:.1f}/10.0, {timer.get_duration_str()})")
                else:
                    self.logger.warning(f"Evaluation failed: {paper_name} ({result.final_score:.1f}/10.0, {timer.get_duration_str()})")
        
        except Exception as e:
            error_msg = f"Knowledge evaluation failed: {e}"
            self.logger.error(f"{error_msg}")
            state["last_error"] = error_msg
            state["evaluation_result"] = {"passed": False, "error": error_msg}
        
        return state
    
    def route_after_evaluation(self, state: LargeScaleWorkflowState) -> str:
        """评估后的路由"""
        evaluation_result = state.get("evaluation_result", {})

        # 确保 evaluation_result 不为 None
        if evaluation_result is None:
            evaluation_result = {}

        if evaluation_result.get("passed"):
            return "generate_qa"
        elif state["current_attempts"] < state["max_attempts"] and self.config['retry']['enable_improvement_suggestions']:
            return "retry"
        else:
            return "finalize"
    
    def decide_retry(self, state: LargeScaleWorkflowState) -> LargeScaleWorkflowState:
        """决定是否重试"""
        state["current_attempts"] += 1
        
        if state["current_attempts"] < state["max_attempts"]:
            state["should_retry"] = True
            self.logger.info(f"Retrying extraction: {state['current_paper']} (attempt {state['current_attempts']}/{state['max_attempts']})")
        else:
            state["should_retry"] = False
            self.logger.warning(f"Maximum retry attempts reached: {state['current_paper']}")
        
        return state
    
    def route_after_retry_decision(self, state: LargeScaleWorkflowState) -> str:
        """重试决定后的路由"""
        if state.get("should_retry", False):
            return "retry_extraction"
        else:
            return "give_up"
    
    def generate_qa(self, state: LargeScaleWorkflowState) -> LargeScaleWorkflowState:
        """生成QA对"""
        paper_name = state["current_paper"]
        extraction_file = state.get("extraction_file")
        
        if not extraction_file:
            state["last_error"] = "No extraction file available for QA generation"
            return state
        
        self.logger.info(f"Generating QA pairs: {paper_name}")
        
        try:
            with Timer(f"QA generation {paper_name}") as timer:
                # 获取QA生成器
                qa_gen = self._get_qa_generator()
                
                # 生成QA对
                qa_gen.run_pipeline(
                    ttl_file_path=extraction_file,
                    max_paths_per_section=self.config['qa_generator']['max_paths_per_section'],
                    max_qa_per_section=self.config['qa_generator']['max_qa_per_section']
                )
                
                # 提取论文基础名称用于QA文件
                paper_base_name = Path(paper_name).stem
                
                # QA文件路径（生成器按此命名模式保存）
                qa_file = str(self.directories['qa'] / f"{paper_base_name}_qa_simplified.json")
                state["qa_file"] = qa_file
                
                self.logger.info(f"QA generation completed: {paper_name} ({timer.get_duration_str()})")
        
        except Exception as e:
            error_msg = f"QA generation failed: {e}"
            self.logger.error(f"{error_msg}")
            state["last_error"] = error_msg
            # QA生成失败不影响论文标记为完成
        
        return state
    
    def finalize_paper(self, state: LargeScaleWorkflowState) -> LargeScaleWorkflowState:
        """完成论文处理"""
        paper_name = state["current_paper"]

        # 确定最终状态
        evaluation_result = state.get("evaluation_result", {})
        last_error = state.get("last_error")

        # 确保 evaluation_result 不为 None
        if evaluation_result is None:
            evaluation_result = {}

        if last_error and not evaluation_result.get("passed"):
            # 论文失败
            state["failed_papers"].append(paper_name)
            self.logger.error(f"Paper failed: {paper_name}")
        else:
            # 论文完成
            state["completed_papers"].append(paper_name)
            self.logger.info(f"Paper completed: {paper_name}")
        
        # 更新统计
        state["batch_statistics"]["processed"] += 1
        
        # 重置当前论文状态
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
    
    def finalize_sub_batch(self, state: LargeScaleWorkflowState) -> LargeScaleWorkflowState:
        """完成子批次处理"""
        self.logger.info(f"Sub-batch completed: {state['sub_batch_id']}")
        
        # 更新统计
        state["batch_statistics"]["end_time"] = datetime.now().isoformat()
        state["batch_statistics"]["completed"] = len(state["completed_papers"])
        state["batch_statistics"]["failed"] = len(state["failed_papers"])
        
        return state


def main():
    """主函数"""
    try:
        # 初始化大规模流水线
        print("Starting large-scale knowledge graph pipeline...")
        pipeline = LargeScaleKGPipeline()
        
        # 运行大规模批处理
        result = pipeline.run_large_scale_batch(parallel=True)
        
        if result["success"]:
            print("Large-scale pipeline completed!")
            print(f"Total: {result['total_completed']}/{result['total_papers']} papers processed successfully")
            return 0
        else:
            print(f"Large-scale pipeline failed: {result['error']}")
            return 1
            
    except Exception as e:
        print(f"Pipeline initialization failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
