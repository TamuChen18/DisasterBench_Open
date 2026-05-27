import gc
from typing import List, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

# NOTE: Import clients lazily inside __init__ to avoid hard dependencies
# (e.g. torch) when using API-based backends like OpenAI/OpenRouter.

class IO_Interface:
    """Input/Output system"""

    def __init__(self, api, model_name, tokenizer_name, generation_kwargs=None) -> None:
        self.api = api
        if api in ["openai"]:
            from interfaces.OpenAI_client import OpenAIClient

            self.client = OpenAIClient(model_name=model_name)
        elif api in ["huggingface"]:
            from interfaces.HuggingFace_client import HuggingFaceClient

            self.client = HuggingFaceClient(model_name, tokenizer_name)
        elif api in ["openrouter"]:
            from interfaces.OpenRouter_client import OpenRouterClient

            self.client = OpenRouterClient(model_name=model_name, tokenizer_name=tokenizer_name)
        elif api in ["vllm"]:
            from interfaces.vLLM_client import VLLMClient

            self.client = VLLMClient(model_name)
        else:
            print("API not supported")
        self.default_generation_kwargs = generation_kwargs        

    def generate(self, model_input: Union[str, List[str]], num_return: int = 1, **generation_kwargs):
        """
        Generates responses from LLM.

        """
        kwargs = {**self.default_generation_kwargs, **generation_kwargs}

        if isinstance(model_input, str):
            if num_return == 1:
                return [self.client.generate(prompt=model_input, **kwargs)]
            else:
                return self.client.generate_n(prompt=model_input, n=num_return, **kwargs)

        elif isinstance(model_input, list):
            results = []

            def worker(prompt):
                if num_return == 1:
                    return self.client.generate(prompt=prompt, **kwargs)
                else:
                    return self.client.generate_n(prompt=prompt, n=num_return, **kwargs)

            # Use ThreadPoolExecutor to parallelize generation
            max_workers = 32
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(worker, prompt): prompt for prompt in model_input}
                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        results.append(f"Error processing prompt '{futures[future]}': {e}")

            return results

        else:
            raise ValueError("model_input must be a string or list of strings.")

    def close(self):
        """
        Gracefully close the underlying model client and free GPU memory.
        """
        try:
            # If the client defines its own close() method (e.g. VLLMClient)
            if hasattr(self.client, "close") and self.api == "huggingface":
                self.client.close()
        except Exception as e:
            print(f"[Warning] Failed to close model client: {e}")
            raise
        
        print("[IO_Interface] Cleanup completed successfully.")
