import os
import time
import json
import urllib.request
import urllib.error

try:
    from openai import APIError, OpenAI
except ModuleNotFoundError:
    APIError = None
    OpenAI = None


def _use_max_completion_tokens(model_name: str) -> bool:
    """Newer OpenAI models (e.g. GPT-5 family) require max_completion_tokens instead of max_tokens."""
    m = (model_name or "").lower()
    if "gpt-5" in m or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return True
    return False


def _normalize_chat_kwargs(model_name: str, generation_kwargs: dict) -> dict:
    """Map max_tokens -> max_completion_tokens; drop unsupported params for newer models."""
    out = dict(generation_kwargs)
    if _use_max_completion_tokens(model_name):
        if "max_tokens" in out and out.get("max_tokens") is not None:
            out["max_completion_tokens"] = out.pop("max_tokens")
        # GPT-5+ responses API style: `stop` is not supported (see invalid_request_error).
        out.pop("stop", None)
    return out


# Chat Completions caps parallel completions per request (e.g. n<=8 on GPT-5.x).
_OPENAI_CHAT_MAX_N = 8


class OpenAIClient:
    """
    A client wrapper for interacting with OpenAI chat models (e.g., GPT-4o).

    This class handles API initialization, prompt generation, retry logic, and 
    supports both single and multiple completions.

    Attributes:
        api_key (str): OpenAI API key.
        model_ckpt (str): The model checkpoint to use (e.g., 'gpt-4o-mini').
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.
        top_k (int): Not currently supported in OpenAI API but included for compatibility.
        top_p (float): Nucleus sampling probability.
        stop (List[str]): Stop sequences for generation.
    """

    def __init__(
        self,
        model_name="gpt-4o-mini",
        tokenizer_name = None
    ):
        """
        Initializes the LLMClient with specified generation parameters.

        Args:
            api_key (str, optional): OpenAI API key. Defaults to environment variable OPENAI_API_KEY.
            model_ckpt (str): Model checkpoint to use.
            max_tokens (int): Max tokens to generate.
            temperature (float): Sampling temperature.
            top_k (int): (Reserved for compatibility) Top-k sampling parameter.
            top_p (float): Nucleus sampling parameter.
            stop (List[str], optional): Stop sequences.
        """

        self.api_key = None or os.getenv("OPENAI_API_KEY")
        
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key) if OpenAI is not None else None
        else:
            raise Exception("OpenAI api key not set. Set API key using export OPENAI_API_KEY=<api key value>")

        self.model_name = model_name
        self._base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    def _http_chat_completions(self, messages, n=1, **kwargs):
        """
        Minimal HTTP fallback for environments without the `openai` Python package.
        Uses Chat Completions endpoint.
        """
        payload = {"model": self.model_name, "messages": messages}
        payload.update(kwargs)
        if n is not None and int(n) != 1:
            payload["n"] = int(n)

        req = urllib.request.Request(
            url=f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        return json.loads(body)

    def generate(self, prompt, **generation_kwargs):
        """
        Generate a single response from the LLM.

        Args:
            prompt (str): User prompt to send to the model.

        Returns:
            str: Model-generated text response.
        """

        messages = [{"role": "user", "content": prompt}]

        kwargs = _normalize_chat_kwargs(self.model_name, generation_kwargs)
        ans, timeout = "", 5
        first_attempt = True
        while not ans:
            try:
                if not first_attempt:
                    time.sleep(timeout)
                first_attempt = False
                if self.client is not None:
                    completion = self.client.chat.completions.create(
                        model=self.model_name, messages=messages, **kwargs
                    )
                    ans = completion.choices[0].message.content
                else:
                    completion = self._http_chat_completions(messages, n=1, **kwargs)
                    ans = completion["choices"][0]["message"]["content"]
            except Exception as e:
                # If openai SDK is available, preserve its bad-request / auth behavior.
                if APIError is not None and isinstance(e, APIError):
                    code = getattr(e, "status_code", None)
                    err_s = str(e).lower()
                    if code == 401:
                        print("OpenAI APIError 401. Check OPENAI_API_KEY. Not retrying.")
                        raise
                    if code == 400 and (
                        "max_completion_tokens" in err_s
                        or "max_tokens" in err_s
                        or "unsupported_parameter" in err_s
                        or "integer_above_max_value" in err_s
                        or "invalid 'n'" in err_s
                    ):
                        print("OpenAI 400 (bad request / unsupported params). Not retrying.")
                        raise
                if isinstance(e, urllib.error.HTTPError):
                    # surface rate-limit/auth errors quickly
                    if e.code in (401, 403, 400):
                        raise
                print(e)
                timeout = min(timeout * 2, 120)
                print(f"Will retry after {timeout} seconds ...")
        return ans

    def _generate_n_chunk(self, messages, chunk_n, kwargs):
        """One Chat Completions call with n=chunk_n (must be <= _OPENAI_CHAT_MAX_N)."""
        ans, timeout = [], 5
        first_attempt = True
        while not ans:
            try:
                if not first_attempt:
                    time.sleep(timeout)
                first_attempt = False
                if self.client is not None:
                    completion = self.client.chat.completions.create(
                        model=self.model_name, messages=messages, n=chunk_n, **kwargs
                    )
                    ans = [choice.message.content for choice in completion.choices]
                else:
                    completion = self._http_chat_completions(messages, n=chunk_n, **kwargs)
                    ans = [c["message"]["content"] for c in completion["choices"]]
            except Exception as e:
                if APIError is not None and isinstance(e, APIError):
                    code = getattr(e, "status_code", None)
                    err_s = str(e).lower()
                    if code == 401:
                        print("OpenAI APIError 401. Check OPENAI_API_KEY. Not retrying.")
                        raise
                    if code == 400 and (
                        "max_completion_tokens" in err_s
                        or "max_tokens" in err_s
                        or "unsupported_parameter" in err_s
                        or "integer_above_max_value" in err_s
                        or "invalid 'n'" in err_s
                    ):
                        print("OpenAI 400 (bad request / unsupported params). Not retrying.")
                        raise
                if isinstance(e, urllib.error.HTTPError):
                    if e.code in (401, 403, 400):
                        raise
                print(e)
                timeout = min(timeout * 2, 120)
                print(f"Will retry after {timeout} seconds ...")
        return ans

    def generate_n(self, prompt, n=1, **generation_kwargs):
        """
        Generate multiple responses from the LLM.

        Args:
            prompt (str): User prompt to send to the model.
            n (int): Number of completions to generate.

        Returns:
            List[str]: List of generated text responses.
        """
        messages = [{"role": "user", "content": prompt}]
        kwargs = _normalize_chat_kwargs(self.model_name, generation_kwargs)
        if n <= _OPENAI_CHAT_MAX_N:
            return self._generate_n_chunk(messages, n, kwargs)
        # e.g. gpt-5.4: n>8 returns 400 — split into multiple requests
        out = []
        remaining = n
        while remaining > 0:
            k = min(remaining, _OPENAI_CHAT_MAX_N)
            out.extend(self._generate_n_chunk(messages, k, kwargs))
            remaining -= k
        return out

    def close(self):
        pass