.PHONY: all latex notebook quarto merge clean

all: latex notebook quarto merge

latex:
	pdflatex -interaction=nonstopmode ./maman11.tex

notebook:
	jupyter nbconvert --to notebook --execute ./BPETokenizer.ipynb --inplace

quarto: notebook
	quarto render BPETokenizer.ipynb --metadata-file pdf-style.yml --to pdf

merge: latex quarto notebook
	python3 -c "\
	from pypdf import PdfWriter; \
	merger = PdfWriter(); \
	merger.append('./maman11.pdf'); \
	merger.append('./BPETokenizer.pdf'); \
	merger.write('merged.pdf'); \
	merger.close()"

clean:
	rm -f merged.pdf BPETokenizer.pdf maman11.pdf ../out/maman11.pdf
	rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz