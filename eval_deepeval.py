import llama_cpp
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

from utils import read_serialized_generated_answer

# this is the json schema that we want to constrain the output to
json_schema = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "verdict": {"type": "string", "enum": ["yes", "no"]},
                    "reason": {"type": "string"},
                },
                "required": ["verdict", "reason"],
            },
        }
    },
    "required": ["verdicts"],
}


# Inherit from DeepEvalBaseLLM for llama.cpp support
class LlamaCppModel(DeepEvalBaseLLM):
    def __init__(self, model_path):
        self.model = llama_cpp.Llama(
            model_path=model_path,
            use_mmap=True,
            verbose=True,
            chat_format="llama-3",
            n_ctx=8092,
            n_gpu_layers=-1,
            verbose=True,
        )
        super().__init__()

    def get_model_name(self):
        return "Meta-Llama-3-8B-Instruct"

    def load_model(self):
        return self.model

    def generate(self, prompt):
        response = self.model.create_chat_completion(
            messages=[{"role": "system", "content": prompt}],
            response_format={
                "type": "json_object",
                "schema": json_schema,
            },
        )
        print(response["choices"][0]["message"]["content"])
        return response["choices"][0]["message"]["content"]

    async def a_generate(self, prompt):
        return self.generate(prompt)


query, documents, answers, staff_answers = read_serialized_generated_answer(
    "generated_data_2024-04-22_17-46-28.json"
)

llama3 = LlamaCppModel(model_path="models/Meta-Llama-3-8B-Instruct-Q8_0.gguf")

# Defining the retrieval metrics
contextual_precision = ContextualPrecisionMetric(
    threshold=0.5, model=llama3, include_reason=True, async_mode=True
)
contextual_recall = ContextualRecallMetric(
    threshold=0.5, model=llama3, include_reason=True, async_mode=True
)
contextual_relevancy = ContextualRelevancyMetric(
    threshold=0.5, model=llama3, include_reason=True, async_mode=True
)

# Defining the generation metrics
# Json constraint might not working for this??
# We need to modify the json schena but in such a way that the schema
# changes based on metric type. How do we do this?
answer_relevancy = AnswerRelevancyMetric(
    threshold=0.5, model=llama3, include_reason=True, async_mode=True
)
faithfullness = FaithfulnessMetric(
    threshold=0.5, model=llama3, include_reason=True, async_mode=True
)


# Evaulates retrieval and generation metrics
def rag_evaluation(query, documents, answers, staff_answers):
    test_cases = []
    for i in range(len(query)):
        test_case = LLMTestCase(
            input=query[i],
            actual_output=answers[i],
            expected_output=staff_answers[i],
            retrieval_context=documents[i],
        )
        test_cases.append(test_case)

    # Evaluate all test cases
    results = evaluate(
        test_cases=test_cases,
        metrics=[
            contextual_precision,
            contextual_recall,
            contextual_relevancy,
            answer_relevancy,
            faithfullness,
        ],
        ignore_errors=True,
        print_results=True,
        run_async=True,
    )
    return results


test_results = rag_evaluation(query, documents, answers, staff_answers)

for result in test_results:
    print(result)
