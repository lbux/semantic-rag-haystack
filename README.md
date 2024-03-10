## Environment Setup Instructions

Follow these steps to set up your development environment:

### 1. Ensure Prerequisites are Installed

Before creating the Conda environment, ensure that the following prerequisites are installed on your system:

- **Make:** Required for building some of the packages. The installation method depends on your operating system:
  - **Linux:** You can typically install `make` via your package manager, for example, `sudo apt-get install make` on Debian/Ubuntu.
  - **macOS:** `make` can be installed with Homebrew using `brew install make`.
  - **Windows:** Consider using a package manager like Chocolatey (`choco install make`) or installing Make for Windows.
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

Finally, install the `llama-cpp-haystack` package:

```bash
pip install llama-cpp-haystack
```

---

### Additional Notes:

- Replace `ENVNAME` with your preferred environment name.
- While using Conda is not required, it is highly recommended for managing dependencies and ensuring consistent environments.

## First Run Instructions

To get started with the project, follow these steps:

1. **Download the Mistral Model:** Run the following command to download the Mistral model from HuggingFace into the `/models` directory.

   ```bash
   python model_download.py
   ```
2. **Prepare Source Documents:** Create a `/SOURCE_DOCUMENTS` folder at the root of your project directory and add your PDF, TXT, and/or MD files that you wish to process.
3. **Ingest Documents into ChromaDB:** Run the following command to ingest the documents in `/SOURCE_DOCUMENTS` using ChromaDB.

   ```bash
   python ingest.py
   ```
4. **Generate Answers from Prompts:** Execute the following command to generate answers from the prompts list in `main.py`. Results will be serialized for further use.

   ```bash
   python main.py
   ```
5. **Evaluate LLM Answers:**
   Finally, run the evaluator script. You'll be prompted to provide your `OPENAI_API_KEY` for this step. The script evaluates LLM answers to prompts, and the results will be serialized.

   ```bash
   python evaluator.py
   ```

## Subsequent Runs

- **Adding Documents:** If you add new documents to `/SOURCE_DOCUMENTS`, you'll need to rerun the `ingest.py` script to process these new documents.
- **After Ingestion:** Following any run of `ingest.py`, you should also rerun `main.py` and `evaluator.py` to ensure that your answers and evaluations are up-to-date with the latest document set.
- **Changes to `main.py`:** If you make modifications only in `main.py`, it necessitates a rerun of `evaluator.py` to evaluate any new or altered prompts.
- **Modifying Evaluation Metrics:** Should there be any changes to the evaluation metric in `evaluator.py`, rerunning `evaluator.py` alone suffices to apply these changes to your evaluation results.
