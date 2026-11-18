.PHONY: all latex notebook quarto merge clean

all: latex notebook quarto merge

latex:
	pdflatex -interaction=nonstopmode ./maman12.tex

notebook:
	jupyter nbconvert --to notebook --execute ./NGramLanguageModel.ipynb --inplace

quarto: notebook
	quarto render NGramLanguageModel.ipynb --metadata-file pdf-style.yml --to pdf

#merge: latex quarto notebook
#	python3 -c "\
#	from pypdf import PdfWriter; \
#	merger = PdfWriter(); \
#	merger.append('./maman11.pdf'); \
#	merger.append('./BPETokenizer.pdf'); \
#	merger.write('merged.pdf'); \
#	merger.close()"

#zip: merge
#	zip maman11.zip BPETokenizer.ipynb ./merged.pdf

clean:
	jupyter nbconvert --clear-output --inplace NGramLanguageModel.ipynb
	rm -f merged.pdf NGramLanguageModel.pdf maman12.pdf
	rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz
	rm *.zip