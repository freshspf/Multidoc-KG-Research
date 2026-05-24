#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to run the Multi-Agent Knowledge Graph Pipeline
"""

import os
import sys
import argparse
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Load environment variables from .env file FIRST
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed, using system environment variables only")

from workflow import MultiAgentKGPipeline
from utils.helpers import validate_environment, load_config


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Multi-Agent Knowledge Graph Pipeline")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume", 
        action="store_true",
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--validate-env", 
        action="store_true",
        help="Validate environment variables only"
    )
    
    args = parser.parse_args()
    
    # Validate environment
    if args.validate_env:
        print("Validating environment variables...")

        # Load configuration first
        try:
            config = load_config(args.config)
        except Exception as e:
            print(f"Failed to load configuration file: {e}")
            config = {}

        validation = validate_environment(config)

        for var, info in validation.items():
            status = "OK" if info['present'] else "MISSING"
            required = "(required)" if info['required'] else "(optional)"
            source = f" ({info['source']})" if info['present'] else ""
            print(f"{status} {var} {required}{source}")

        # Check if all required variables are present
        missing_required = [var for var, info in validation.items() if info['required'] and not info['present']]
        if missing_required:
            print(f"\nMissing required API configuration: {', '.join(missing_required)}")
            print("Please set them in your config.yaml file or environment variables")
            return 1
        else:
            print("\nAll required API configuration is present")
            return 0
    
    try:
        # Load configuration first
        try:
            config = load_config(args.config)
        except Exception as e:
            print(f"Failed to load configuration file: {e}")
            return 1

        # Validate environment before starting
        validation = validate_environment(config)
        missing_required = [var for var, info in validation.items() if info['required'] and not info['present']]
        if missing_required:
            print(f" Missing required API configuration: {', '.join(missing_required)}")
            print("Please set them in your config.yaml file or environment variables:")
            for var in missing_required:
                print(f"  {var}")
            return 1
        
        print("Starting Multi-Agent Knowledge Graph Pipeline...")
        print(f"Configuration: {args.config}")
        
        # Initialize and run pipeline
        pipeline = MultiAgentKGPipeline(config_path=args.config)
        result = pipeline.run_batch(resume_from_checkpoint=args.resume)
        
        if result["success"]:
            print("\nPipeline completed successfully!")
            
            # Print summary statistics
            stats = result.get("batch_statistics", {})
            if stats:
                print("\n Summary Statistics:")
                print(f"   Total papers: {stats.get('total_papers', 0)}")
                print(f"   Processed: {stats.get('processed', 0)}")
                print(f"   Completed: {stats.get('completed', 0)}")
                print(f"   Failed: {stats.get('failed', 0)}")
                
                if stats.get('start_time') and stats.get('end_time'):
                    from datetime import datetime
                    start = datetime.fromisoformat(stats['start_time'])
                    end = datetime.fromisoformat(stats['end_time'])
                    duration = (end - start).total_seconds()
                    print(f"   Duration: {duration:.1f}s")
            
            return 0
        else:
            print(f"\nPipeline failed: {result['error']}")
            if result.get('traceback'):
                print("\n Error details:")
                print(result['traceback'])
            return 1
            
    except Exception as e:
        print(f" Pipeline initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())