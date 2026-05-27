import os
import time

from openai import APIError, OpenAI


class OpenRouterClient:
    """
    OpenRouter client via OpenAI-compatible Chat Completions API.

    Env vars:
      - OPENROUTER_API_KEY: required
      - OPENROUTER_BASE_URL: optional (default https://openrouter.ai/api/v1)
    """

    def __init__(self, model_name: str, tokenizer_name=None):
        del tokenizer_name
        if not model_name:
            raise ValueError("model_name is required (OpenRouter model id).")

        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OpenRouter_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set in environment/.env (also tried OpenRouter_API_KEY)")

        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt, **generation_kwargs):
        return self.generate_n(prompt, n=1, **generation_kwargs)[0]

    def generate_n(self, prompt, n=1, **generation_kwargs):
        n = max(1, int(n))
        max_tokens = generation_kwargs.get("max_tokens", 1024)
        temperature = generation_kwargs.get("temperature", 0.7)
        top_p = generation_kwargs.get("top_p", 0.95)
        stop = generation_kwargs.get("stop", None)
        response_format = generation_kwargs.get("response_format")

        create_kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "n": n,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop,
        }
        if response_format is not None:
            create_kwargs["response_format"] = response_format

        ans = None
        timeout = 2
        while ans is None:
            try:
                completion = self.client.chat.completions.create(**create_kwargs)
                choices = getattr(completion, "choices", None) or []
                if not choices:
                    raise RuntimeError("OpenRouter returned empty choices (choices is None or [])")
                outs = []
                for c in choices:
                    msg = getattr(c, "message", None)
                    if msg is None:
                        outs.append("")
                        continue
                    content = getattr(msg, "content", None)
                    # Some reasoning-capable models may return text primarily in `reasoning`
                    # when `content` is empty in non-stream mode.
                    if content is None or content == "":
                        content = getattr(msg, "reasoning", None)
                    outs.append(content or "")

                if all(o == "" for o in outs):
                    try:
                        finish_reasons = [getattr(c, "finish_reason", None) for c in choices]
                        usage = getattr(completion, "usage", None)
                        print(
                            f"OpenRouter empty text output for model={self.model_name}, "
                            f"finish_reasons={finish_reasons}, usage={usage}"
                        )
                    except Exception:
                        pass
                # Some providers may return fewer than requested `n`; keep only real outputs.
                return outs
            except APIError as e:
                code = getattr(e, "status_code", None)
                err_s = str(e).lower()
                # Quota / billing / auth: retrying will not help; avoid infinite 120s loops.
                if code == 401:
                    print("OpenRouter APIError 401 (unauthorized). Check OPENROUTER_API_KEY. Not retrying.")
                    raise
                if code == 403 and (
                    "limit" in err_s
                    or "quota" in err_s
                    or "exceeded" in err_s
                    or "billing" in err_s
                ):
                    print(
                        "OpenRouter 403: key limit / quota exceeded. "
                        "Add credits at https://openrouter.ai/settings/keys — not retrying."
                    )
                    raise RuntimeError(
                        "OpenRouter key limit exceeded (403). Stop the job and fix billing/limits."
                    ) from e
                print(e)
                timeout = min(timeout * 2, 120)
                print(f"Will retry after {timeout} seconds ...")
                time.sleep(timeout)
            except Exception as e:
                print(e)
                timeout = min(timeout * 2, 120)
                print(f"Will retry after {timeout} seconds ...")
                time.sleep(timeout)

        return ["" for _ in range(n)]

