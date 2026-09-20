"""Make the repo root importable so `bonsai_forensics` resolves from source."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
