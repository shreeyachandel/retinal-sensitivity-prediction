# Public dissertation source

This is the buildable LaTeX source for the public MSc dissertation. Two retinal
quality-control figures were replaced with synthetic schematics; clinical data
are not included.

Build from this directory:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The tracked final document is `../../output/pdf/public-thesis.pdf`.
