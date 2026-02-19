#!/usr/bin/env python3
"""Standalone launcher for the Overcooked V3 Level Editor.

This script doesn't require full JaxMARL installation, only pygame and numpy.
"""

import sys
import importlib.util
from pathlib import Path

# Load layout_editor_v3.py directly without importing through jaxmarl package
editor_path = Path(__file__).parent / "jaxmarl" / "tools" / "layout_editor_v3.py"
spec = importlib.util.spec_from_file_location("layout_editor_v3", editor_path)
layout_editor = importlib.util.module_from_spec(spec)
sys.modules["layout_editor_v3"] = layout_editor

# Add jaxmarl to path for relative imports within the editor
sys.path.insert(0, str(Path(__file__).parent))

spec.loader.exec_module(layout_editor)
layout_editor.main()
