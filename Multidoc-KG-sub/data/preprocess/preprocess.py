#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compatibility wrapper for the unified PDF preprocessing entrypoint.

`bookmark_based_splitter.py` is now the single maintained implementation.
This module is kept so older commands/imports do not break immediately.
"""

from bookmark_based_splitter import BookmarkBasedSplitter, BookmarkInfo, ProcessingStats, Section, main

__all__ = [
    "BookmarkBasedSplitter",
    "BookmarkInfo",
    "ProcessingStats",
    "Section",
    "main",
]


if __name__ == "__main__":
    main()
