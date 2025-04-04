# zebras-stitching
Exploring how to stitch zebra data

# Using UV with pyproject.toml

This project uses a `pyproject.toml` file to define dependencies and project configuration. Here's how to use UV with this project:

## Installation with UV

1. Install UV if you don't already have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create a virtual environment and install all dependencies from pyproject.toml:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```