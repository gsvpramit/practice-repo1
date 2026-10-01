# practice-repo1
For training purposes

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` covers the scientific stack, classical ML, deep learning
(PyTorch), Hugging Face Transformers, LLM SDKs, LangChain, LangGraph, and
vector stores for RAG.

## Run

```bash
python main.py "What is gradient descent?"
```

Set `ANTHROPIC_API_KEY` (or put it in a `.env` file) to get a real answer from
Claude; without it, the LangGraph workflow runs in offline mode.
