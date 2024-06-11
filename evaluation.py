from haystack.components.evaluators import FaithfulnessEvaluator

from utils.serializing import (
    read_serialized_generated_answer,
    serialize_evaluation_results,
)

queries, documents, answers, staff_answers = read_serialized_generated_answer(
    "generated_data_2024-06-10_20-24-21.json"
)


evaluator = FaithfulnessEvaluator(
    api="llama_cpp",
    api_params={
        "model": "models/Meta-Llama-3-8B-Instruct-Q6_K.gguf",
        "model_kwargs": {"n_gpu_layers": -1, "n_predict": -1},
    },
)

all_results = []
counter = 0

for query, document, answer in zip(queries, documents, answers):
    if counter > 3:
        break
    try:
        result = evaluator.run(
            questions=[query], contexts=[document], predicted_answers=[answer]
        )
        all_results.append(result)
    except ValueError as e:
        print(f"Skipping due to error: {e}")
        continue
    counter += 1

print(all_results)

serialize_evaluation_results(all_results)
