#!/bin/sh

INFILE="${1:-deepseek-new-baseline.ipynb}"
OUTFILE="${2:-deepseekmath_nb.py}"

jupytext --to py:percent --opt comment_magics=0 --opt cell_metadata_filter=-_cell_guid,-_uuid,-papermill "$INFILE" -o "$OUTFILE"

#jupytext --to ipynb deepseekmath_nb.py -o baseline.ipynb
