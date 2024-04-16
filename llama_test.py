from llama_cpp import Llama
llm = Llama(
      model_path="models/mistral-7b-instruct-v0.2.Q8_0.gguf",
      # n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
      # n_ctx=2048, # Uncomment to increase the context window
)