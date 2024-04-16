from evaluate import evaluator
from datasets import load_dataset
from utils import (
    read_serialized_generated_answer,
    read_input_json,
    # metric_to_params,
    # serialize_evaluation_results,
)
# Use a pipeline as a high-level helper
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

#pipe = pipeline("text-generation", model="mistral-7b-instruct-v0.2.Q8_0.gguf")
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

task_evaluator = evaluator("question-answering")

queries, documents, answers = read_serialized_generated_answer()

dataset = load_dataset("json", data_files="summer21.json", split="train")

eval_results = task_evaluator.compute(
    model_or_pipeline= model,
    data= dataset,
    # question_column= queries,
    # context_column= documents,
    # label_column= answers,
    strategy="bootstrap",
    n_resamples=30
)

