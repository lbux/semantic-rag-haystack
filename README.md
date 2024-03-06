
## Environment Setup Instructions

Follow these steps to set up your development environment:

### 1. Ensure Prerequisites are Installed

Before creating the Conda environment, ensure that the following prerequisites are installed on your system:

- **Make:** Required for building some of the packages. The installation method depends on your operating system:

  - On Linux, you can typically install `make` via your package manager, for example, `sudo apt-get install make` on Debian/Ubuntu.
  - On macOS, `make` can be installed with brew using `brew install make`.
  - On Windows, you might consider using a package manager like Chocolatey (`choco install make`) or installing .
- **NVIDIA CUDA Toolkit 11.8:** Required for GPU support and compiling CUDA packages. Installation methods vary by operating system:

  - **Linux:** Follow the [official Linux installation guide](https://developer.nvidia.com/cuda-downloads?target_os=Linux) provided by NVIDIA, selecting the appropriate version and distribution.
  - **macOS:** CUDA support for macOS is limited and may not be available for newer versions. Check the [CUDA Toolkit archive](https://developer.nvidia.com/nvidia-cuda-toolkit-developer-tools-mac-hosts) for compatibility.
  - **Windows:** Use the [official Windows installation guide](https://developer.nvidia.com/cuda-downloads?target_os=Windows) to download and install the toolkit suitable for your system.

### 2. Create Conda Environment

Create a Conda environment named `ENVNAME` with Python 3.10. Open your terminal or command prompt and run:

```bash
conda create -n ENVNAME python=3.10
```

### 3. Activate Conda Environment

Activate the newly created Conda environment:

```bash
conda activate ENVNAME
```

### 4. Install Required Python Packages

Install the required Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 5. Install `llama-cpp-python` with Custom CMake Arguments

The installation of `llama-cpp-python` requires setting the `CMAKE_ARGS` environment variable before installation. The steps vary slightly between Linux/Mac and Windows.

#### For Linux and Mac:

Run the following command in your terminal:

```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
```

#### For Windows:

Open PowerShell and execute the following commands:

```powershell
$env:CMAKE_ARGS = "-DLLAMA_CUBLAS=on"
pip install llama-cpp-python
```

### 6. Install `llama-cpp-haystack`

Finally, install the `llama-cpp-haystack ` package:

```bash
pip install python-cpp-haystack
```

---

### Additional Notes:

- Replace `ENVNAME` with preferred environment name.
- Using conda is not required but highly recommended. Most issues can be resolved by

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
