# Secure Memory Validation Framework

For evaluating candidate long-term memories against memory-poisoning risks. It combines candidate-only control checks, semantic retrieval, compositional and contradiction analysis, a knowledge graph, configurable scoring, quarantine and manual review.

## System requirements

- Windows 10 or 11 (64-bit)
- Python 3.12 (64-bit)
- Internet access during the first setup and first model download
- Approximately 3 GB of free disk space for Python packages, PyTorch, and model files
- A modern browser

The local Ollama LLM is optional. The deterministic analysis works when Ollama is unavailable.

## Quick start on Windows

1. Download or clone this repository.
2. Double-click `setup_windows.bat` once. This creates `venv` and installs `requirements.txt`.
3. Double-click `run_windows.bat`.
4. Open the local URL printed by Streamlit, normally `http://localhost:8501`.

`run_windows.bat` automatically launches the setup script when the virtual environment does not exist.

The first application run downloads `sentence-transformers/all-MiniLM-L6-v2`. This download is cached in the user's Hugging Face cache for later runs.

## Optional local LLM

Install [Ollama](https://ollama.com/) and run:

```powershell
ollama pull qwen2.5:0.5b
```

Keep Ollama running while using the application. Without it, the UI reports **Local LLM: Offline** and continues with deterministic scoring. The LLM is advisory and does not override the deterministic decision.

## Run from PowerShell

```powershell
cd path\to\2024MT13029
.\setup_windows.bat
.\run_windows.bat
```