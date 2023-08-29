import requests
import json

from langchain.llms.base import LLM


class CustomLLM(LLM):
    model_name: str
    completions_url: str = "http://localhost:8000/v1/chat/completions"
    temperature: float = 0.
    max_tokens: int = 1024

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop=None) -> str:
        r = requests.post(
            self.completions_url,
            json={
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stop": stop,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            },
        )
        result = r.json()
        try:
            return result["choices"][0]["message"]["content"]
        except:
            raise RuntimeError(result)

    async def _acall(self, prompt: str, stop=None) -> str:
        return self._call(prompt, stop)


class CustomLLMV3(LLM):
    model_name: str
    completions_url: str = "http://localhost:8004/v1/completions"
    temperature: float = 0.
    max_tokens: int = 1024

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop=None) -> str:
        r = requests.post(
            self.completions_url,
            json={
                "model": self.model_name,
                "prompt": json.loads(prompt),
                "stop": "\n\n",
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            },
        )
        result = r.json()
        if result.get("object") == "error":
            raise RuntimeError(result.get("message"))
        else:
            return result["choices"][0]["text"]

    async def _acall(self, prompt: str, stop=None) -> str:
        return self._call(prompt, stop)
