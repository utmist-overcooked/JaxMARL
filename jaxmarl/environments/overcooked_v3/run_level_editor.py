#!/usr/bin/env python3
"""Launcher script for the Overcooked V3 Level Editor."""

import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from jaxmarl.tools.layout_editor_v3 import main

if __name__ == "__main__":
    main()
