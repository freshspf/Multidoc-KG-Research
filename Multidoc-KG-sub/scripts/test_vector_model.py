"""
Lightweight script to verify that a sentence-transformers embedding model can be loaded.

Usage examples:
  python scripts/test_vector_model.py
  python scripts/test_vector_model.py --model BAAI/bge-m3
  python scripts/test_vector_model.py --model BAAI/bge-m3 --retries 5 --delay 3
"""
import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Test loading a sentence-transformers model.")
    parser.add_argument("--model", default="BAAI/bge-m3", help="Model name or local path")
    parser.add_argument("--retries", type=int, default=3, help="Number of load attempts")
    parser.add_argument("--delay", type=float, default=2.0, help="Base delay between retries in seconds")
    parser.add_argument(
        "--text",
        default="hepatocellular carcinoma biomarker",
        help="Short probe text to encode after the model loads",
    )
    args = parser.parse_args()

    print(f"[test_vector_model] model={args.model}")
    print(f"[test_vector_model] cwd={Path.cwd()}")

    last_error = None
    for attempt in range(1, max(1, args.retries) + 1):
        try:
            print(f"[test_vector_model] attempt {attempt}/{args.retries}: importing sentence_transformers")
            from sentence_transformers import SentenceTransformer

            print(f"[test_vector_model] attempt {attempt}/{args.retries}: loading model")
            model = SentenceTransformer(args.model)

            dim = model.get_sentence_embedding_dimension()
            print(f"[test_vector_model] loaded successfully, embedding_dim={dim}")

            vector = model.encode([args.text], convert_to_numpy=True)
            print(f"[test_vector_model] encode ok, shape={tuple(vector.shape)}")
            print("[test_vector_model] success")
            return 0

        except KeyboardInterrupt:
            print("[test_vector_model] interrupted by user")
            return 130
        except Exception as exc:
            last_error = exc
            print(f"[test_vector_model] failed: {type(exc).__name__}: {exc}")
            if attempt >= args.retries:
                break

            sleep_s = args.delay * (2 ** (attempt - 1))
            print(f"[test_vector_model] retrying in {sleep_s:.1f}s...")
            time.sleep(sleep_s)

    print("[test_vector_model] giving up")
    if last_error is not None:
        print(f"[test_vector_model] final_error={type(last_error).__name__}: {last_error}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
