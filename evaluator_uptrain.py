from uptrain import EvalLLM, Evals, ResponseMatching, Settings

from utils import read_serialized_generated_answer, serialize_evaluation_results

queries, documents, answers, staff_answers = read_serialized_generated_answer(
    "documents\generation_output\input.json"
)

data = [
    {
        "question": queries[3],
        "response": answers[3],
        "ground_truth": staff_answers[3],
    }
]

settings = Settings(model="ollama/llama3")
eval_llm = EvalLLM(settings=settings)

results = eval_llm.evaluate(
    project_name="local test",
    data=data,
    checks=[ResponseMatching(method="llm")],
)

print(results)
