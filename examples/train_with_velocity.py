#!/usr/bin/env python3
"""
Example: Training scIDiff with RNA velocity as biological prior

This script demonstrates how to train the scIDiff drift field model with
RNA velocity integration. The velocity acts as a biological compass that
guides the Schrödinger Bridge toward trajectories that are not just
mathematically optimal but also biologically plausible.

Usage:
    python train_with_velocity.py --h5ad data.h5ad --use-velocity-prior

For more options, run:
    python train_with_velocity.py --help
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scqdiff.pipeline.train_from_anndata import main

if __name__ == '__main__':
    main()
