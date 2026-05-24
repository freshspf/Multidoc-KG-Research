#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-hop QA评估工具 - 仅场景B版本
评估大模型在仅问答对场景下的表现
支持并行处理和随机选择功能

使用示例：

1. 基本使用（评估1000篇论文）:
python3 qa_evaluation_scenario_b_only.py \
  --qa-dir /path/to/qa/data \
  --max-workers 8 \
  --num-papers 1000 \
  --api-url https://api.example.com/v1/chat/completions \
  --api-key YOUR_API_KEY

2. 随机选择5%的文件进行评估:
python3 qa_evaluation_scenario_b_only.py \
  --qa-dir /path/to/qa/data \
  --max-workers 8 \
  --percentage 0.05 \
  --seed 42 \
  --api-url https://api.example.com/v1/chat/completions \
  --api-key YOUR_API_KEY \
  --results-file evaluation_results.json

3. 使用环境变量设置API密钥:
export API_KEY="your_api_key_here"
python3 qa_evaluation_scenario_b_only.py \
  --qa-dir /path/to/qa/data \
  --api-url https://api.example.com/v1/chat/completions
"""

import os
import json
import random
import re
import requests
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import partial

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 模型配置
AVAILABLE_MODELS = {
    "AutoQG-20k": {
        "llama3.1-70b": "llama-3.1-70b-instruct",
        "qwen2.5-72b": "qwen2.5-72b-instruct",
        "doubao-pro": "Doubao-pro-32k",
        "o4-mini": "o4-mini-2025-04-16",
        "llama-4-maverick": "meta-llama/llama-4-maverick",
        "ernie-3.5": "ERNIE-3.5-8K",
        "gpt-5-mini": "gpt-5-mini",
        "gpt-5-nano": "gpt-5-nano"
    }
}


class LLMInterface:
    """LLM接口调用器 - 支持并行调用"""
    
    def __init__(self, api_url: str, api_key: str, max_workers: int = 5):
        self.api_url = api_url
        self.api_key = api_key
        self.max_workers = max_workers
        self._lock = threading.Lock()
    
    def call_model(self, model_id: str, prompt: str, max_tokens: int = 200, timeout: int = 30) -> Optional[str]:
        """
        调用LLM模型
        
        Returns:
            模型回答文本，失败返回None
        """
        payload = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    message = result['choices'][0].get('message', {})
                    content = message.get('content')
                    
                    if content is None:
                        return None
                    
                    return content.strip() if content else ""
            
            return None
            
        except Exception as e:
            logger.debug(f"模型调用失败 {model_id}: {e}")
            return None
    
    def call_models_parallel(self, model_prompts: List[Tuple[str, str, str]], max_tokens: int = 200, timeout: int = 30) -> Dict[str, Optional[str]]:
        """
        并行调用多个模型
        
        Args:
            model_prompts: List of (model_name, model_id, prompt)
            max_tokens: 最大token数
            timeout: 超时时间
            
        Returns:
            Dict[model_name, response_text]
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_model = {
                executor.submit(self.call_model, model_id, prompt, max_tokens, timeout): model_name
                for model_name, model_id, prompt in model_prompts
            }
            
            # 收集结果
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result = future.result()
                    results[model_name] = result
                except Exception as e:
                    logger.debug(f"并行调用失败 {model_name}: {e}")
                    results[model_name] = None
        
        return results


class AnswerExtractor:
    """答案提取器"""
    
    @staticmethod
    def extract_answer(response: str) -> Optional[str]:
        """
        从模型回答中提取A/B/C/D答案
        
        Args:
            response: 模型回答文本
            
        Returns:
            提取的答案选项，失败返回None
        """
        if not response:
            return None
        
        # 常见的答案模式
        patterns = [
            r'The correct answer is boxed\s*\{([ABCD])\}',  # 新格式
            r'boxed\s*\{([ABCD])\}',  # 简化格式
            r'答案是\s*([ABCD])',
            r'选择\s*([ABCD])', 
            r'正确答案是\s*([ABCD])',
            r'答案：\s*([ABCD])',
            r'选项\s*([ABCD])',
            r'^([ABCD])$',  # 单独的字母
            r'^([ABCD])\.',  # A. B. C. D.
            r'\b([ABCD])\b'  # 任何地方的单独字母
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        return None


class QAEvaluator:
    """QA评估器 - 仅场景B版本"""
    
    def __init__(self, qa_dir: str, api_url: str, api_key: str, 
                 max_workers: int = 5, enable_parallel: bool = True,
                 resume_from_checkpoint: bool = True, results_file: str = "qa_evaluation_results.json"):
        self.qa_dir = Path(qa_dir)
        self.llm_interface = LLMInterface(api_url, api_key, max_workers)
        self.answer_extractor = AnswerExtractor()
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers
        self.resume_from_checkpoint = resume_from_checkpoint
        self.results_file = results_file
        
        self.evaluation_results = {}
        self.total_questions = 0
        self.selected_papers = []
        self.completed_papers = set()  # 已完成评估的论文ID集合
        self._results_lock = threading.Lock()
        self._progress_lock = threading.Lock()
        
        # 加载已有结果（如果存在）
        if self.resume_from_checkpoint:
            self._load_existing_results()
    
    def _load_existing_results(self):
        """加载已有的评估结果，提取已完成的论文ID"""
        results_path = Path(self.results_file)
        
        if not results_path.exists():
            logger.info(f"结果文件不存在: {self.results_file}，将从头开始评估")
            return
        
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            # 从detailed_results中提取已完成的论文ID
            if 'detailed_results' in existing_data:
                for result in existing_data['detailed_results']:
                    paper_id = result.get('paper_id')
                    if paper_id:
                        self.completed_papers.add(str(paper_id))
                        self.total_questions += 1
            
            logger.info(f"加载已有结果: {len(self.completed_papers)} 个论文已完成评估，共 {self.total_questions} 个问答对")
            logger.info(f"已完成的论文ID: {sorted(list(self.completed_papers))[:10]}{'...' if len(self.completed_papers) > 10 else ''}")
            
        except Exception as e:
            logger.warning(f"加载已有结果失败: {e}，将从头开始评估")
            self.completed_papers = set()
            self.total_questions = 0
    
    def select_random_papers(self, num_papers: int = 10, percentage: float = None, seed: int = 42) -> List[str]:
        """随机选择论文，排除已完成评估的论文，支持百分比选择"""
        qa_files = list(self.qa_dir.glob("paper_*_QA_pair.json"))
        
        # 提取所有可用的paper_id
        available_paper_ids = []
        for file in qa_files:
            match = re.search(r'paper_(\d+)_QA_pair\.json', file.name)
            if match:
                paper_id = match.group(1)
                # 只选择未完成评估的论文
                if paper_id not in self.completed_papers:
                    available_paper_ids.append(paper_id)
        
        logger.info(f"总共 {len(qa_files)} 个QA文件，其中 {len(self.completed_papers)} 个已完成评估")
        logger.info(f"剩余 {len(available_paper_ids)} 个论文待评估")
        
        if len(available_paper_ids) == 0:
            logger.warning("所有论文都已完成评估！")
            return []
        
        # 设置随机种子
        random.seed(seed)
        
        # 根据百分比或数量选择论文
        if percentage is not None:
            select_count = max(1, int(len(available_paper_ids) * percentage))
            logger.info(f"根据百分比 {percentage*100:.1f}% 选择 {select_count} 篇论文")
        else:
            select_count = min(num_papers, len(available_paper_ids))
            if len(available_paper_ids) < num_papers:
                logger.warning(f"待评估文件数量({len(available_paper_ids)})少于需求数量({num_papers})")
                select_count = len(available_paper_ids)
        
        # 随机选择未完成的论文
        selected_paper_ids = random.sample(available_paper_ids, select_count)
        
        logger.info(f"随机选择了 {len(selected_paper_ids)} 篇待评估论文 (随机种子: {seed})")
        return selected_paper_ids
    
    def create_prompt_scenario_b(self, question: str, options: Dict[str, str]) -> str:
        """创建场景B的prompt - 测试模型在没有任何上下文时的表现"""
        options_text = "\n".join([f"{key}. {value}" for key, value in options.items()])
        
        prompt = f"""Please answer the following multiple-choice question based on your general knowledge. Each question has only 1 correct option.
Format your response as follows:
The correct answer is boxed {{insert answer here}}.

Examples:
The correct answer is boxed {{C}}

Question: {question}

Options:
{options_text}

Answer:"""
        return prompt
    
    def evaluate_single_qa(self, qa_item: Dict, paper_id: str) -> Dict[str, Any]:
        """评估单个问答对 - 仅场景B"""
        question = qa_item.get('question', '')
        options = qa_item.get('options', {})
        correct_answer = qa_item.get('correct_answer', '')
        section = qa_item.get('section', '')
        
        # 创建场景B的prompt
        prompt_b = self.create_prompt_scenario_b(question, options)
        
        qa_result = {
            'paper_id': paper_id,
            'section': section,
            'question': question[:100] + "..." if len(question) > 100 else question,
            'correct_answer': correct_answer,
            'scenario_b': {}
        }
        
        # 获取所有模型ID
        all_models = {}
        for group in AVAILABLE_MODELS.values():
            all_models.update(group)
        
        if self.enable_parallel:
            # 并行调用所有模型
            model_prompts = []
            
            for model_name, model_id in all_models.items():
                model_prompts.append((f"{model_name}_scenario_b", model_id, prompt_b))
            
            # 并行调用
            responses = self.llm_interface.call_models_parallel(model_prompts)
            
            # 处理结果
            for model_name, model_id in all_models.items():
                response_b = responses.get(f"{model_name}_scenario_b")
                extracted_b = self.answer_extractor.extract_answer(response_b) if response_b else None
                qa_result['scenario_b'][model_name] = {
                    'response': response_b,
                    'extracted_answer': extracted_b,
                    'is_correct': extracted_b == correct_answer if extracted_b else False
                }
        else:
            # 串行调用
            for model_name, model_id in all_models.items():
                response_b = self.llm_interface.call_model(model_id, prompt_b)
                extracted_b = self.answer_extractor.extract_answer(response_b) if response_b else None
                
                qa_result['scenario_b'][model_name] = {
                    'response': response_b,
                    'extracted_answer': extracted_b,
                    'is_correct': extracted_b == correct_answer if extracted_b else False
                }
                
                # 避免请求过于频繁
                time.sleep(0.5)
        
        return qa_result
    
    def print_intermediate_results(self, all_results: List[Dict], completed_papers: int):
        """打印中间评估结果"""
        if not all_results:
            return
        
        # 获取所有模型名称
        all_models = {}
        for group in AVAILABLE_MODELS.values():
            all_models.update(group)
        
        model_stats = {}
        
        for model_name in all_models.keys():
            model_stats[model_name] = {
                'scenario_b': {'correct': 0, 'total': 0, 'accuracy': 0.0}
            }
        
        # 统计当前结果
        for result in all_results:
            for model_name in all_models.keys():
                if model_name in result.get('scenario_b', {}):
                    model_stats[model_name]['scenario_b']['total'] += 1
                    if result['scenario_b'][model_name]['is_correct']:
                        model_stats[model_name]['scenario_b']['correct'] += 1
        
        # 计算准确率
        for model_name in model_stats:
            stats = model_stats[model_name]['scenario_b']
            if stats['total'] > 0:
                stats['accuracy'] = stats['correct'] / stats['total']
        
        # 输出中间结果
        logger.info(f"已完成 {completed_papers} 篇论文，总问答对数: {len(all_results)}")
        logger.info("当前准确率统计:")
        logger.info("场景B: 仅问答对")
        logger.info(f"{'模型名称':<25} {'场景B准确率':<12} {'问答对数'}")
        logger.info("-" * 50)
        
        sorted_models = sorted(model_stats.items(), 
                             key=lambda x: x[1]['scenario_b']['accuracy'], 
                             reverse=True)
        
        for model_name, stats in sorted_models:
            scenario_b_acc = stats['scenario_b']['accuracy'] * 100
            total_qa = stats['scenario_b']['total']
            logger.info(f"{model_name:<25} {scenario_b_acc:>8.1f}%    {total_qa:>6}")
    
    def evaluate_papers(self, paper_ids: List[str]):
        """评估选定的论文 - 支持并行评估"""
        logger.info(f"开始评估 {len(paper_ids)} 篇论文...")
        logger.info(f"并行模式: {'启用' if self.enable_parallel else '禁用'}")
        if self.enable_parallel:
            logger.info(f"最大并行线程数: {self.max_workers}")
        
        if self.enable_parallel:
            # 并行评估模式
            return self._evaluate_papers_parallel(paper_ids)
        else:
            # 串行评估模式
            return self._evaluate_papers_sequential(paper_ids)
    
    def _evaluate_papers_sequential(self, paper_ids: List[str]):
        """串行评估论文"""
        all_results = []
        
        for i, paper_id in enumerate(paper_ids, 1):
            logger.info(f"评估论文 {i}/{len(paper_ids)}: paper_{paper_id}")
            
            # 加载QA文件
            qa_file = self.qa_dir / f"paper_{paper_id}_QA_pair.json"
            
            if not qa_file.exists():
                logger.warning(f"QA文件不存在: {qa_file.name}")
                continue
            
            try:
                with open(qa_file, 'r', encoding='utf-8') as f:
                    qa_data = json.load(f)
                
                logger.info(f"  包含 {len(qa_data)} 个问答对")
                
                # 评估所有问答对
                for j, qa_item in enumerate(qa_data, 1):
                    logger.info(f"  评估问答 {j}/{len(qa_data)}")
                    result = self.evaluate_single_qa(qa_item, paper_id)
                    all_results.append(result)
                    self.total_questions += 1
                
                # 每5篇论文输出一次中间结果
                if i % 5 == 0:
                    logger.info("=" * 70)
                    logger.info(f"中间结果 - 已完成 {i}/{len(paper_ids)} 篇论文")
                    self.print_intermediate_results(all_results, i)
                    logger.info("=" * 70)
                
            except Exception as e:
                logger.error(f"处理QA文件失败 {qa_file.name}: {e}")
        
        return all_results
    
    def _evaluate_papers_parallel(self, paper_ids: List[str]):
        """并行评估论文"""
        all_results = []
        completed_papers = 0
        
        # 准备所有QA任务
        qa_tasks = []
        for paper_id in paper_ids:
            qa_file = self.qa_dir / f"paper_{paper_id}_QA_pair.json"
            
            if not qa_file.exists():
                logger.warning(f"QA文件不存在: {qa_file.name}")
                continue
            
            try:
                with open(qa_file, 'r', encoding='utf-8') as f:
                    qa_data = json.load(f)
                
                for qa_item in qa_data:
                    qa_tasks.append((qa_item, paper_id))
                    
            except Exception as e:
                logger.error(f"加载QA文件失败 {qa_file.name}: {e}")
        
        logger.info(f"总共 {len(qa_tasks)} 个问答对待评估")
        
        # 使用ThreadPoolExecutor并行处理QA任务
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_qa = {
                executor.submit(self.evaluate_single_qa, qa_item, paper_id): (qa_item, paper_id)
                for qa_item, paper_id in qa_tasks
            }
            
            # 收集结果
            completed_count = 0
            for future in as_completed(future_to_qa):
                qa_item, paper_id = future_to_qa[future]
                try:
                    result = future.result()
                    
                    with self._results_lock:
                        all_results.append(result)
                        self.total_questions += 1
                        completed_count += 1
                    
                    # 线程安全的进度显示
                    with self._progress_lock:
                        if completed_count % 10 == 0 or completed_count == len(qa_tasks):
                            logger.info(f"已完成 {completed_count}/{len(qa_tasks)} 个问答对")
                            
                        # 每完成50个问答对显示一次中间结果
                        if completed_count % 50 == 0:
                            logger.info("=" * 70)
                            logger.info(f"中间结果 - 已完成 {completed_count}/{len(qa_tasks)} 个问答对")
                            self.print_intermediate_results(all_results, completed_count // 10)
                            logger.info("=" * 70)
                    
                except Exception as e:
                    logger.error(f"评估问答对失败 paper_{paper_id}: {e}")
        
        return all_results
    
    def aggregate_results(self, all_results: List[Dict]) -> Dict[str, Any]:
        """汇总评估结果"""
        logger.info("汇总评估结果...")
        
        # 获取所有模型名称
        all_models = {}
        for group in AVAILABLE_MODELS.values():
            all_models.update(group)
        
        model_stats = {}
        
        for model_name in all_models.keys():
            model_stats[model_name] = {
                'scenario_b': {'correct': 0, 'total': 0, 'accuracy': 0.0}
            }
        
        # 统计结果
        for result in all_results:
            for model_name in all_models.keys():
                if model_name in result.get('scenario_b', {}):
                    model_stats[model_name]['scenario_b']['total'] += 1
                    if result['scenario_b'][model_name]['is_correct']:
                        model_stats[model_name]['scenario_b']['correct'] += 1
        
        # 计算准确率
        for model_name in model_stats:
            stats = model_stats[model_name]['scenario_b']
            if stats['total'] > 0:
                stats['accuracy'] = stats['correct'] / stats['total']
        
        summary = {
            'evaluation_summary': {
                'total_papers': len(self.selected_papers),
                'total_questions': self.total_questions,
                'models_tested': list(all_models.keys()),
                'scenarios': ['scenario_b']
            },
            'results_by_model': model_stats,
            'detailed_results': all_results
        }
        
        return summary
    
    def print_summary(self, summary: Dict[str, Any]):
        """打印评估摘要"""
        logger.info("=" * 80)
        logger.info("评估结果摘要 - 仅场景B")
        logger.info("=" * 80)
        
        eval_summary = summary['evaluation_summary']
        logger.info(f"评估论文数: {eval_summary['total_papers']}")
        logger.info(f"总问答对数: {eval_summary['total_questions']}")
        logger.info(f"测试模型数: {len(eval_summary['models_tested'])}")
        
        logger.info("\n按模型汇总准确率:")
        results = summary['results_by_model']
        
        logger.info("场景B: 仅问答对")
        logger.info("-" * 50)
        
        sorted_models = sorted(results.items(), 
                             key=lambda x: x[1]['scenario_b']['accuracy'], 
                             reverse=True)
        
        logger.info(f"{'模型名称':<25} {'场景B准确率':<12} {'问答对数'}")
        logger.info("-" * 50)
        
        for model_name, stats in sorted_models:
            scenario_b_acc = stats['scenario_b']['accuracy'] * 100
            total_qa = stats['scenario_b']['total']
            logger.info(f"{model_name:<25} {scenario_b_acc:>8.1f}%    {total_qa:>6}")
    
    def save_results(self, summary: Dict[str, Any], output_file: str = "qa_evaluation_results.json"):
        """保存评估结果，支持增量保存"""
        # 如果启用断点续传且已有结果文件，需要合并结果
        if self.resume_from_checkpoint and Path(output_file).exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                # 合并detailed_results，去重
                existing_results = existing_data.get('detailed_results', [])
                new_results = summary.get('detailed_results', [])
                
                # 创建已有结果的paper_id集合，用于去重
                existing_paper_qa_set = set()
                for result in existing_results:
                    paper_id = result.get('paper_id')
                    question = result.get('question', '')
                    existing_paper_qa_set.add(f"{paper_id}_{hash(question)}")
                
                # 只添加新的结果
                merged_results = existing_results[:]
                for result in new_results:
                    paper_id = result.get('paper_id')
                    question = result.get('question', '')
                    key = f"{paper_id}_{hash(question)}"
                    if key not in existing_paper_qa_set:
                        merged_results.append(result)
                
                # 更新汇总统计
                summary['detailed_results'] = merged_results
                summary['evaluation_summary']['total_questions'] = len(merged_results)
                
                logger.info(f"合并结果: 已有 {len(existing_results)} 个，新增 {len(merged_results) - len(existing_results)} 个")
                
            except Exception as e:
                logger.warning(f"合并已有结果失败: {e}，将覆盖保存")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"详细结果已保存到: {output_file}")
    
    def run_evaluation(self, num_papers: int = 10, percentage: float = None, seed: int = 42):
        """运行完整评估流程"""
        logger.info("=" * 80)
        logger.info("Multi-hop QA评估工具 - 仅场景B版本")
        logger.info("模式: 仅场景B评估（仅问答对）")
        logger.info(f"断点续传: {'启用' if self.resume_from_checkpoint else '禁用'}")
        if self.resume_from_checkpoint:
            logger.info(f"结果文件: {self.results_file}")
        logger.info(f"并行模式: {'启用' if self.enable_parallel else '禁用'}")
        if self.enable_parallel:
            logger.info(f"最大并行线程数: {self.max_workers}")
        logger.info("=" * 80)
        
        # 随机选择论文
        self.selected_papers = self.select_random_papers(num_papers, percentage, seed)
        
        if not self.selected_papers:
            logger.error("没有选择到有效的论文文件")
            return
        
        # 评估论文
        all_results = self.evaluate_papers(self.selected_papers)
        
        # 汇总结果
        summary = self.aggregate_results(all_results)
        
        # 打印摘要
        self.print_summary(summary)
        
        # 保存结果
        self.save_results(summary, self.results_file)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-hop QA评估工具 - 仅场景B版本')
    parser.add_argument('--qa-dir', 
                       required=True,
                       help='问答对目录路径')
    parser.add_argument('--num-papers',
                       type=int,
                       default=1000,
                       help='评估的论文数量')
    parser.add_argument('--api-url',
                       required=True,
                       help='API接口地址')
    parser.add_argument('--api-key',
                       help='API密钥（也可通过环境变量API_KEY设置）')
    parser.add_argument('--max-workers',
                       type=int,
                       default=5,
                       help='并行评估的最大线程数')
    parser.add_argument('--disable-parallel',
                       action='store_true',
                       help='禁用并行评估，使用串行模式')
    parser.add_argument('--disable-resume',
                       action='store_true',
                       help='禁用断点续传，从头开始评估')
    parser.add_argument('--results-file',
                       default='qa_evaluation_results.json',
                       help='结果保存文件路径')
    parser.add_argument('--percentage',
                       type=float,
                       default=None,
                       help='随机选择文件的百分比 (0.01-1.0, 例如: 0.05 表示5%%)')
    parser.add_argument('--seed',
                       type=int,
                       default=42,
                       help='随机种子 (默认: 42)')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.percentage is not None and args.num_papers != 1000:
        logger.warning("同时指定了百分比和论文数量，将优先使用百分比选择")
    
    if args.percentage is not None:
        if not (0.01 <= args.percentage <= 1.0):
            logger.error("百分比必须在0.01到1.0之间")
            exit(1)
    
    # 获取API密钥
    api_key = args.api_key
    if not api_key:
        api_key = os.getenv('API_KEY')
        if not api_key:
            logger.error("API密钥未提供！请通过 --api-key 参数或 API_KEY 环境变量设置")
            exit(1)
    
    # 设置随机种子以便复现结果
    random.seed(args.seed)
    
    # 创建评估器
    evaluator = QAEvaluator(args.qa_dir, args.api_url, api_key, 
                           max_workers=args.max_workers,
                           enable_parallel=not args.disable_parallel,
                           resume_from_checkpoint=not args.disable_resume,
                           results_file=args.results_file)
    
    # 运行评估
    evaluator.run_evaluation(num_papers=args.num_papers, percentage=args.percentage, seed=args.seed)


if __name__ == "__main__":
    main()
