#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to analyze experiment statistics and generate a report
"""

import json
import os
from pathlib import Path
from collections import Counter


def analyze_outputs():
    """Analyze the outputs directory and generate statistics"""
    
    # Load the latest large scale report
    report_files = list(Path("outputs").glob("large_scale_report_*.json"))
    if not report_files:
        print("No large scale report found")
        return
    
    # Get the most recent report
    latest_report = sorted(report_files, reverse=True)[0]
    
    with open(latest_report, 'r') as f:
        report_data = json.load(f)
    
    # Analyze QA files
    qa_dir = Path("getKG-schema/multi_hop_qa")
    if not qa_dir.exists():
        print("QA directory not found")
        return
    
    qa_files = list(qa_dir.glob("*_qa_simplified.json"))
    
    total_questions = 0
    papers_with_qa = 0
    question_distribution = []
    
    for qa_file in qa_files:
        try:
            with open(qa_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    question_count = len(data)
                    total_questions += question_count
                    if question_count > 0:
                        papers_with_qa += 1
                    question_distribution.append(question_count)
        except Exception as e:
            print(f"Error reading {qa_file}: {e}")
    
    # Generate statistics report
    stats_report = {
        "experiment_overview": {
            "total_papers_processed": report_data["total_papers"],
            "successful_extractions": report_data["total_completed"],
            "failed_extractions": report_data["total_failed"],
            "success_rate": f"{report_data['success_rate']:.2f}%",
            "processing_duration_hours": f"{report_data['duration_seconds']/3600:.2f}",
            "start_time": report_data["start_time"],
            "end_time": report_data["end_time"]
        },
        "qa_generation": {
            "papers_with_qa": papers_with_qa,
            "total_questions_generated": total_questions,
            "average_questions_per_paper": f"{total_questions/max(papers_with_qa, 1):.2f}",
            "max_questions_in_single_paper": max(question_distribution) if question_distribution else 0,
            "min_questions_in_single_paper": min(question_distribution) if question_distribution else 0
        },
        "batch_processing": {
            "total_batches": report_data["total_batches"],
            "batch_size": report_data["batch_size"],
            "parallel_processing": report_data["parallel_processing"],
            "max_concurrent_batches": report_data["max_concurrent"]
        }
    }
    
    # Save statistics report
    stats_file = Path("experiment_statistics_report.json")
    with open(stats_file, 'w') as f:
        json.dump(stats_report, f, indent=2)
    
    # Print summary
    print("实验统计摘要")
    print("=" * 50)
    print(f"总处理论文数: {stats_report['experiment_overview']['total_papers_processed']}")
    print(f"成功提取: {stats_report['experiment_overview']['successful_extractions']}")
    print(f"失败提取: {stats_report['experiment_overview']['failed_extractions']}")
    print(f"成功率: {stats_report['experiment_overview']['success_rate']}")
    print(f"处理时长: {stats_report['experiment_overview']['processing_duration_hours']} 小时")
    print()
    print("问答生成统计:")
    print(f"  生成问答的论文数: {stats_report['qa_generation']['papers_with_qa']}")
    print(f"  总问题数: {stats_report['qa_generation']['total_questions_generated']}")
    print(f"  平均每篇论文问题数: {stats_report['qa_generation']['average_questions_per_paper']}")
    print(f"  单篇论文最多问题数: {stats_report['qa_generation']['max_questions_in_single_paper']}")
    print(f"  单篇论文最少问题数: {stats_report['qa_generation']['min_questions_in_single_paper']}")
    print()
    print(f"批处理信息:")
    print(f"  总批次数: {stats_report['batch_processing']['total_batches']}")
    print(f"  批次大小: {stats_report['batch_processing']['batch_size']}")
    print(f"  并行处理: {stats_report['batch_processing']['parallel_processing']}")
    print(f"  最大并发批次数: {stats_report['batch_processing']['max_concurrent_batches']}")
    
    return stats_report


if __name__ == "__main__":
    analyze_outputs()