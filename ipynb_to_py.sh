#!/bin/sh

jupytext --to py:percent --opt comment_magics=0 --opt cell_metadata_filter=-_cell_guid,-_uuid,-papermill deepseek-new-baseline.ipynb -o deepseekmath_nb.py

#jupytext --to ipynb deepseekmath_nb.py -o baseline.ipynb
