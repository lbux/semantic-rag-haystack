from uptrain import EvalLLM, Evals, Settings
import json
from utils import read_serialized_generated_answer, serialize_evaluation_results

queries, documents, answers, staff_answers = read_serialized_generated_answer("generated_data_2024-04-15_22-15-39.json")

data = [
    {
        "question": queries[1],
        "context": documents[1],
        "response": answers[1],
    }
]

settings = Settings(model='ollama/wizardlm2')
eval_llm = EvalLLM(settings=settings)

results = eval_llm.evaluate(project_name = "uptrain local test", data=data, checks=[Evals.CONTEXT_RELEVANCE])

print(results)