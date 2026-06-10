# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a [Quarto](https://quarto.org/) website containing learning materials (slide decks and notebooks) for [skrub](https://skrub-data.org/), a Python library for machine learning with dataframes. Content is published automatically to GitHub Pages on push to `main`.

## Commands

```bash
# Render the full site
pixi run render-all
# or equivalently
quarto render

# Render a single file
quarto render pages/slides/skrub-intro/index.qmd

# Preview the site with live reload
quarto preview

# Convert Python scripts to notebooks (uses jupytext + nbconvert)
make all

# Convert a single .py to .ipynb
jupytext --to notebook pages/notebooks/skrub-intro/notebook.py
```

Environment is managed with [pixi](https://pixi.sh). Run `pixi install` to set up the environment. There is no test suite — this is a documentation/content project.

## Architecture

### Content locations

- [pages/slides/](pages/slides/) — RevealJS slide decks, one subdirectory per talk (each with an `index.qmd`)
- [pages/notebooks/](pages/notebooks/) — Jupyter-based tutorials
- [pages/slides/index_slides.qmd](pages/slides/index_slides.qmd) and [pages/notebooks/index_notebooks.qmd](pages/notebooks/index_notebooks.qmd) — auto-generated listing pages

### Modular includes

Slide decks are built from reusable Markdown snippets in [includes/](includes/). Individual `.qmd` files use Quarto's `{{< include >}}` directive to compose slides from shared sections:

```
includes/
├── talk-sections/    # Reusable talk sections (intro, teaser, getting-involved, ...)
├── data_ops/         # DataOps / expressions feature docs
├── encoders/         # Encoder feature docs
└── preparation/      # Data preparation docs
```

When editing a shared section, all talks that include it are affected. Check which talks use an include before editing.

### Adding a new talk

1. Create a subdirectory under `pages/slides/<talk-name>/`
2. Add an `index.qmd` with RevealJS format and required YAML frontmatter (see existing talks for the pattern)
3. The talk will appear automatically in the listing page

### Build / deploy

GitHub Actions ([.github/workflows/publish.yml](.github/workflows/publish.yml)) runs `quarto render` on every push to `main` and publishes the output to the `gh-pages` branch (hosted at `skrub-data.org/skrub-materials/`).

Quarto's freeze feature ([`_freeze/`](_freeze/)) caches rendered notebook outputs so unchanged notebooks are not re-executed on each build.
