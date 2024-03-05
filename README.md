Environment setup:
  1. It is recommended to use a conda environment with python 3.x (tbd)
  2. Within the environment, run `pip install haystack-ai chroma-haystack uptrain-haystack pypdf markdown-it-py mdit_plain`
  3. For Windows, install `llama-cpp-python` as follows: `$env:CMAKE_ARGS="-DLLAMA_CUBLAS=on"` `pip install llama-cpp-python`
  4. Lastly, run `pip install llama-cpp-haystack`

First Run:
  1. Run `python model_download.py` to download the Mistral model from HuggingFace into `/models`
  2. Create `/SOURCE_DOCUMENTS` folder and add pdf, txt, and/or md files
  3. Run `python ingest.py` to ingest documents in `/SOURCE_DOCUMENTS` using ChromaDB
  4. Run `python main.py` to generate answers from the `prompts` list in `main.py`. Results will be serialized.
  5. Run `python evaluator.py` and provide `OPENAI_API_KEY` to evaluate LLM answers to `prompts`. Results will be serialized.

Subsequent Runs:
  1. Adding documents to `/SOURCE_DOCUMENTS` necessitates rerunning `ingest.py`
  2. Reruns of `ingest.py` should also be followed with runs of `main.py` and `evaluator.py`
  3. Changes in `main.py` only necessitates running `evaluator.py`
  4. Changing metric in `evaluator.py` only necessitates running `evaluator.py`
