# Documentation Build Guide

This directory contains the source files for the laser-measles documentation.

## Quick Start

### HTML Documentation (using tox)

```bash
# Build with default Python (3.12)
tox -e docs

# Build with specific Python version
TOXPYTHON=python3.11 tox -e docs
```

### PDF Documentation

PDF generation requires Docker (for LaTeX):

```bash
# 1. Build HTML first (to generate intermediates)
tox -e docs

# 2. Generate LaTeX source
source .tox/docs/bin/activate
sphinx-build -b latex docs dist/latex

# 3. Compile to PDF using Docker
docker run --rm -v "$(pwd)/dist/latex:/work" -w /work texlive/texlive:latest \
    sh -c "pdflatex -interaction=nonstopmode laser-measles.tex && \
           pdflatex -interaction=nonstopmode laser-measles.tex"
```

## Manual Build (without tox)

### Prerequisites

- Python 3.10, 3.11, or 3.12
- Docker (for PDF generation)
- pandoc (automatically downloaded by build script)

### HTML Build Steps

```bash
# 1. Create and activate virtual environment
python3.11 -m venv .venv-docs
source .venv-docs/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -e .
pip install -r docs/requirements.txt
pip install jupytext

# 3. Download and install pandoc
curl -L https://github.com/jgm/pandoc/releases/download/3.7.0.2/pandoc-3.7.0.2-linux-amd64.tar.gz -o pandoc.tar.gz
tar -xzf pandoc.tar.gz
cp pandoc-3.7.0.2/bin/pandoc .venv-docs/bin/

# 4. Convert tutorial Python files to notebooks
cd docs/tutorials
bash create_ipynb.sh
cd ../..

# 5. Generate API documentation
sphinx-apidoc -f -o docs/reference --module-first src/laser/measles

# 6. Build HTML documentation
sphinx-build -E -b html docs dist/docs
```

Output: `dist/docs/index.html`

### PDF Build Steps

```bash
# 1. Activate virtual environment (from HTML build above)
source .venv-docs/bin/activate

# 2. Build LaTeX source
sphinx-build -b latex docs dist/latex

# 3. Compile to PDF (requires Docker)
# Run 2-3 times to resolve cross-references
docker run --rm -v "$(pwd)/dist/latex:/work" -w /work texlive/texlive:latest \
    pdflatex -interaction=nonstopmode laser-measles.tex

docker run --rm -v "$(pwd)/dist/latex:/work" -w /work texlive/texlive:latest \
    pdflatex -interaction=nonstopmode laser-measles.tex
```

Output: `dist/latex/laser-measles.pdf`

**Note:** The first Docker run downloads the TeX Live image (~4GB), which takes several minutes.

## Directory Structure

```
docs/
├── api/                    # API documentation templates
├── images/                 # Images used in documentation
├── reference/              # Auto-generated API reference (created by sphinx-apidoc)
├── tutorials/              # Tutorial notebooks (*.ipynb generated from *.py)
│   ├── *.py               # Tutorial source files (tracked in git)
│   ├── *.ipynb            # Generated notebooks (not tracked in git)
│   └── create_ipynb.sh    # Conversion script using jupytext
├── _static/               # Static files (CSS, JS, etc.)
├── _templates/            # Custom Sphinx templates
├── conf.py                # Sphinx configuration
├── index.rst              # Documentation home page
├── requirements.txt       # Python dependencies for docs build
└── README.md              # This file
```

## Troubleshooting

### Import Errors During Build

If you see `ModuleNotFoundError: No module named 'laser_core'`:

```bash
# Ensure laser-core version is compatible
pip install "laser-core<=0.6"
```

### Unicode Errors in PDF

Some Unicode characters (like δ) may cause LaTeX warnings but won't prevent PDF generation. The build uses `-interaction=nonstopmode` to continue through errors.

### Notebook Conversion Fails

If `jupytext` is not found:

```bash
pip install jupytext
```

### Docker Permission Issues

If you get permission errors with Docker:

```bash
# Copy PDF to a location with correct permissions
cp dist/latex/laser-measles.pdf dist/docs/laser-measles.pdf
```

## CI/CD

The documentation is automatically built and deployed via:
- **ReadTheDocs**: Builds on every push to main
- **GitHub Actions**: Runs `tox -e docs` in CI pipeline

## Additional Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [nbsphinx - Jupyter Integration](https://nbsphinx.readthedocs.io/)
- [PyData Sphinx Theme](https://pydata-sphinx-theme.readthedocs.io/)
