#!/bin/bash
echo "Showing clean code structure (excluding datasets, dumps, heavy folders)..."
tree --prune -I '.git|__pycache__|dumps|dataset|rendered_video|mesh_snapshot|lfs|objects|cache'