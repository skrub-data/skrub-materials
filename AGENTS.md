# Skrub Learning Materials - Agent Guide

This is a [Quarto](https://quarto.org/) website hosting learning materials for [Skrub](https://skrub-data.org/stable/), a Python library for data preparation and machine learning preprocessing.

## Quick Start

**Build/Render the site:**
```bash
quarto render                    # Render the entire site
make render-all                  # Same as above (via pixi task)
```

**Environment setup:**
```bash
pixi install                     # Install dependencies
pixi shell                       # Activate environment
```

**Run specific tasks:**
```bash
pixi run render-all              # Render all content
```

## Project Structure

### Main Directory Layout
```
pages/
  slides/              # Presentation slides (revealjs format)
    index_slides.qmd   # Listing of all slide presentations
    [event-name]/      # Each event/talk has its own directory
      slides.qmd       # Main slide file
      style.css        # Optional: event-specific styling
      [resources]/     # Event-specific data files, images, etc.
  
  notebooks/           # Jupyter notebooks converted from .py or .ipynb
    index_notebooks.qmd
    [topic]/           # Organized by topic

includes/              # Reusable markdown snippets (shared across presentations)
  data_ops/            # Data operations content
  encoders/            # Feature encoding content
  preparation/         # Data preparation content
  talk-sections/       # Reusable talk sections

resources/             # Global assets: images, data files, etc.
  csv-*.csv            # Sample datasets
  parallel_coordinates_hgbr.json
  dataop_report/       # Pre-generated HTML reports

_freeze/               # Quarto cache (auto-generated)
_quarto.yml            # Main Quarto configuration
pixi.toml              # Pixi environment definition
```

## Working with Slides

### Creating a New Slide Presentation

1. **Create a new directory** in `pages/slides/[event-name]/`
2. **Create `slides.qmd`** with this template:

```yaml
---
title: "Your Presentation Title"
subtitle: "Subtitle here"
date: 2026-06-10
author: "Your Name"
institute: ""
format: 
    revealjs:
        slide-number: c/t
        show-slide-number: all
        preview-links: auto
        embed-resources: false
        transition: slide
        theme: simple
        logo: /images/skrub.svg
        css: style.css
        footer: "https://skrub-data.org/skrub-materials/"
incremental: false
params: 
    version: "base"
---
```

### Reusing Content with `includes/`

Instead of duplicating content across presentations, use Quarto's include shortcode:

```qmd
{{< include /includes/preparation/_data_cleaning_skrub.md >}}
```

**Available include categories:**
- `/includes/talk-sections/` - Introductions, conclusions, topic summaries
- `/includes/preparation/` - Data cleaning and exploration content
- `/includes/encoders/` - Feature encoding techniques
- `/includes/data_ops/` - Data operations and transformations

### Slide Conventions

- **Slides in revealjs format** - Use `##` for slide headers, `#` for sections
- **Auto-animate slides** - Add `{auto-animate="true"}` to slide class for transitions
- **Code execution** - Use ` ```{.python}` blocks for executable Python code
- **Responsive images** - Reference global images with `/images/filename`
- **Section markers** - Use `#` for major sections that appear in outline
- **Smaller text** - Add `.smaller` class for slide headers with lots of text

### Adding Event-Specific Resources

Store event resources in `pages/slides/[event-name]/`:
- Data files, images, JSON data
- Custom CSS in `style.css` (in addition to global theme)
- Subdirectories for organized resources

## Working with Notebooks

Notebooks are stored in `pages/notebooks/[topic]/` and can be:
- `.ipynb` files (Jupyter notebooks)
- `.py` files (converted to notebooks via `jupytext`)

## Key Conventions

### Imports and Dependencies
- **Python version:** 3.10+
- **Key libraries:**
  - `skrub` >= 0.6.2 (data preprocessing)
  - `polars` (dataframe library - used alongside pandas)
  - `plotly` (interactive plots)
  - `jupyter` (for notebook execution)

### Content Organization Principles
1. **Modular includes** - Extract reusable sections to `includes/[category]/` for sharing across presentations
2. **Event directories** - Each presentation is self-contained in `pages/slides/[event-name]/`
3. **Semantic naming** - Include files prefixed with `_` followed by descriptive names
4. **No duplication** - If content appears in multiple presentations, move it to `includes/`

### YAML Front Matter Patterns
- **`title-block-banner: true`** - Shows title as a prominent banner
- **`params: version: "base"`** - Used for conditional content rendering
- **`incremental: false`** - Quarto revealjs animations (set per presentation)

## Common Tasks

| Task | Command |
|------|---------|
| Render entire site | `quarto render` |
| Preview specific presentation | Navigate to `pages/slides/[event-name]/` and use Quarto VS Code extension |
| Check dependencies | `pixi list` |
| Add a package | Edit `pixi.toml`, then `pixi install` |
| Update include file | Edit file in `includes/[category]/_filename.md` |
| Create new slide deck | Create directory, add `slides.qmd` with template |

## Dependency Management

Dependencies are managed via `pixi.toml`:

```toml
[dependencies]
skrub = ">=0.6.2,<0.7"           # Main library
jupyter = ">=1.1.1,<2"           # Notebook support
polars = ">=1.35.1,<2"           # DataFrame library
pyarrow = ">=22.0.0,<23"         # Data format support
plotly = ">=6.5.0,<7"            # Interactive plots
ipython = ">=9.7.0,<10"          # IPython kernel
```

**Version Pinning Strategy:** Strict upper bounds (`<X.Y`) to ensure reproducibility and test against known compatible versions.

## Important Notes

- Content may become **outdated** as Skrub evolves - refer to [official documentation](https://skrub-data.org/stable/reference/index.html) for up-to-date API information
- Rendered output is in `_freeze/` (auto-generated, do not edit directly)
- The website is hosted at [skrub-data.org/skrub-materials/](https://skrub-data.org/skrub-materials/)
- For executable examples, refer to the [main gallery](https://skrub-data.org/stable/auto_examples/index.html)

## File Types by Location

| Location | File Type | Purpose |
|----------|-----------|---------|
| `pages/slides/[event]/` | `.qmd` | Presentation source (revealjs) |
| `pages/notebooks/[topic]/` | `.ipynb` or `.py` | Notebook content |
| `includes/[category]/` | `.md` | Reusable markdown snippets |
| `resources/` | `.csv`, `.json`, `.html` | Data and assets for all presentations |
| Root | `_quarto.yml` | Site-wide Quarto configuration |
| Root | `pixi.toml` | Environment and task definitions |

---

**Version:** 0.1.0 | **Last updated:** June 2026
