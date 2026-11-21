# ==============================================================================
# Configuration Parameters
# ==============================================================================
NAME       := NGramLanguageModel
ASSIGNMENT := maman12

.PHONY: all latex notebook quarto merge clean

all: latex notebook quarto

latex:
	pdflatex -interaction=nonstopmode ./$(ASSIGNMENT).tex

notebook:
	jupyter nbconvert --to notebook --execute ./$(NAME).ipynb --inplace

quarto: notebook
	quarto render $(NAME).ipynb --metadata-file pdf-style.yml --to pdf

merge: latex quarto notebook
	python3 -c "\
	from pypdf import PdfWriter; \
	merger = PdfWriter(); \
	merger.append('./$(ASSIGNMENT).pdf'); \
	merger.append('./$(NAME).pdf'); \
	merger.write('merged.pdf'); \
	merger.close()"

#zip: merge
#	zip $(ASSIGNMENT).zip BPETokenizer.ipynb ./merged.pdf

clean:
	jupyter nbconvert --clear-output --inplace $(NAME).ipynb
	rm -f merged.pdf $(ASSIGNMENT).pdf $(NAME).pdf
	rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz
	rm -f *.zip