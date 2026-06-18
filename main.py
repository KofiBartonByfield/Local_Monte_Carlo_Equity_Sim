"""
main.py — entry point for the VaR Engine desktop app.
Run:  python main.py
"""
import sys
import os

# Ensure local packages resolve correctly when run from project root
sys.path.insert(0, os.path.dirname(__file__))

from gui.app import run

if __name__ == "__main__":
    run()
