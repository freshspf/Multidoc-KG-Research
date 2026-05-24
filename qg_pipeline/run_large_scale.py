#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大规模知识图谱流水线运行脚本
用于处理1000+论文的批处理任务
"""

import argparse
import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, continue without it

from large_scale_workflow import LargeScaleKGPipeline

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Large-scale knowledge graph pipeline")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Configuration file path (default: config.yaml)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        help="Number of papers per sub-batch (override config file setting)"
    )
    parser.add_argument(
        "--max-concurrent", 
        type=int, 
        help="Maximum concurrent batch count (override config file setting)"
    )
    parser.add_argument(
        "--serial", 
        action="store_true",
        help="Use serial processing mode (instead of parallel)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Dry run mode, only show batch division"
    )
    
    args = parser.parse_args()
    
    try:
        print( "Initializing large-scale knowledge graph pipeline...")
        print(f"Configuration file: {args.config}")
        
        # 初始化流水线
        pipeline = LargeScaleKGPipeline(args.config)
        
        # 覆盖配置文件设置
        if args.batch_size:
            pipeline.batch_size = args.batch_size
            print(f" Overriding batch size: {args.batch_size}")
            
        if args.max_concurrent:
            pipeline.max_concurrent = args.max_concurrent
            print(f"Overriding maximum concurrent count: {args.max_concurrent}")
        
        if args.dry_run:
            print("\nDry run mode - analyzing batch division...")
            
            # 获取论文列表
            from utils.helpers import get_input_papers
            papers = get_input_papers(pipeline.config)
            
            if not papers:
                print("No input papers found")
                return 1
            
            print(f"Found {len(papers)} papers")
            
            # 显示批次划分
            batches = pipeline.split_papers_into_batches(papers)
            
            print(f"\nBatch division scheme:")
            print(f"   Total papers: {len(papers)}")
            print(f"   Batch size: {pipeline.batch_size}")
            print(f"   Total batches: {len(batches)}")
            print(f"   Maximum concurrent: {pipeline.max_concurrent}")
            print(f"   Processing mode: {'parallel' if not args.serial else 'serial'}")
            
            print(f"\nDetailed batch information:")
            for i, batch in enumerate(batches):
                print(f"   Batch {i+1}: {len(batch)} papers")
                if len(batch) <= 5:
                    for j, paper in enumerate(batch):
                        print(f"     {j+1}. {paper}")
                else:
                    for j in range(3):
                        print(f"     {j+1}. {batch[j]}")
                    print(f"     ... {len(batch)-3} more")
            
            # 估算处理时间
            avg_time_per_paper = 120  # 假设每篇论文2分钟
            if not args.serial:
                estimated_time = (len(papers) / pipeline.max_concurrent * avg_time_per_paper) / 60
            else:
                estimated_time = (len(papers) * avg_time_per_paper) / 60
            
            print(f"\nEstimated processing time: {estimated_time:.1f} minutes")
            print("Dry run completed")
            
            return 0
        
        # 运行大规模批处理
        print(f"\nStarting large-scale batch processing...")
        print(f"Current settings:")
        print(f"   Batch size: {pipeline.batch_size}")
        print(f"   Maximum concurrent: {pipeline.max_concurrent}")
        print(f"   Processing mode: {'serial' if args.serial else 'parallel'}")
        
        result = pipeline.run_large_scale_batch(parallel=not args.serial)
        
        if result["success"]:
            print("\nLarge-scale pipeline processing completed!")
            print("\nFinal statistics:")
            print(f"   Total papers: {result['total_papers']}")
            print(f"   Processing success: {result['total_completed']}")
            print(f"   Processing failed: {result['total_failed']}")
            print(f"   Success rate: {result['success_rate']:.1f}%")
            print(f"   Total time: {result['duration_seconds']:.1f} seconds ({result['duration_seconds']/60:.1f} minutes)")
            print(f"   Average per paper: {result['duration_seconds']/result['total_papers']:.1f} seconds")
            
            if result['total_failed'] > 0:
                print(f"\nFailed papers:")
                for paper in result['failed_papers']:
                    print(f"   - {paper}")
            
            return 0
        else:
            print(f"\nLarge-scale pipeline failed: {result['error']}")
            return 1
            
    except KeyboardInterrupt:
        print("\nUser interrupted processing")
        return 1
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())