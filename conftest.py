"""
conftest.py (project root)
==========================
Pytest configuration: add project root to sys.path so that
`from src.xxx import ...` works in all tests.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
