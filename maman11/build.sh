pdflatex  ./maman11.tex

jupyter nbconvert --to notebook --execute ./BPETokenizer.ipynb --inplace

quarto render BPETokenizer.ipynb --metadata-file pdf-style.yml --to pdf

python3 -c "
from pypdf import PdfWriter

merger = PdfWriter()
merger.append('../out/maman11.pdf')
merger.append('BPETokenizer.pdf')
merger.write('merged.pdf')
merger.close()
"
