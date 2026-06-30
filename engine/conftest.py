"""pytest 路径引导：把 engine/ 根目录加入 sys.path，使 `from schema import ...` 可用。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
