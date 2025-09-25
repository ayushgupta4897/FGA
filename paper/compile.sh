#!/bin/bash

# Compile the FGA paper for arXiv submission
echo "Compiling FGA paper..."

# First pass
pdflatex fga_arxiv_paper.tex

# Run bibtex if needed
# bibtex fga_arxiv_paper

# Second pass for references
pdflatex fga_arxiv_paper.tex

# Third pass for final formatting
pdflatex fga_arxiv_paper.tex

echo "✓ Compilation complete!"
echo "Output: fga_arxiv_paper.pdf"

# Clean up auxiliary files
rm -f *.aux *.log *.out *.toc *.bbl *.blg

echo "✓ Cleaned up auxiliary files"
