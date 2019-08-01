# bsuite experiment report

This is a folder that maintains the latex template for auto-generating a bsuite report appendix, suitable for conference submission.
To do this:
- Fill in bsuite_preamble.tex with relevant links to colab/plots.
- Write description of agents/algorithms in bsuite_appendix.tex together with some commentary on results.
- use \input{} or copy/paste the bsuite_preamble.tex before your \begin{document} and bsuite_appendix.tex inside your document.
- You can find examples of using the bsuite appendix with ICLR, ICML, NeurIPS templates as well as standalone pdf generation in the subfolders here.

