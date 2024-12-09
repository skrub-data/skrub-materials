scripts := $(wildcard *.py)
notebooks := $(patsubst %.py,_build/%.ipynb,$(scripts))
notebooks-nbconvert := $(patsubst %.py,_build/%-nbconvert.html,$(scripts))
notebooks-quarto := $(patsubst %.py,_build/%-quarto.html,$(scripts))


.PHONY: all clean notebooks notebooks-nbconvert notebooks-quarto

all: notebooks

notebooks: $(notebooks)

notebooks-nbconvert: $(notebooks-nbconvert)

notebooks-quarto: $(notebooks-quarto)

_build/%.ipynb: %.py
	mkdir -p _build
	jupytext $< --output $@

%-quarto.html: %.ipynb
	mkdir -p _build
	cd _build && quarto render $(notdir $<) --to html -o $(notdir $@)

%-nbconvert.html: %.ipynb
	mkdir -p _build
	cd _build && jupyter nbconvert $(notdir $<) --execute --to html --output $(notdir $@)

clean:
	rm -rf _build
