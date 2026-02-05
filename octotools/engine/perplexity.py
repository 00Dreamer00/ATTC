try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install the openai package by running `pip install openai`, and add 'DEEPSEEK_API_KEY' to your environment variables.")

import os
import platformdirs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from typing import List, Union
from .base import EngineLM, CachedEngine


class ChatPerplexity(EngineLM, CachedEngine):
    def __init__(
        self,
        model_string,
        use_cache: bool=False,
        is_multimodal: bool=False):

        self.model_string = model_string
        self.use_cache = use_cache
        self.is_multimodal = is_multimodal

        if self.use_cache:
            root = platformdirs.user_cache_dir("octotools")
            cache_path = os.path.join(root, f"cache_perplexity_{model_string}.db")
            super().__init__(cache_path=cache_path)

        if os.getenv("PERPLEXITY_API_KEY") is None:
            raise ValueError("Please set the PERPLEXITY_API_KEY environment variable.")
        
        print(f"Use perplexity {model_string} backend")
        
        self.client = OpenAI(
            api_key=os.getenv("PERPLEXITY_API_KEY"),
            base_url="https://api.perplexity.ai"
        )

    @retry(
    wait=wait_random_exponential(multiplier=1.5, min=3, max=8),  # perplexity rate limit: 50 req per min  
    stop=stop_after_attempt(5),
    )
    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt=None, **kwargs):
        try:
            if isinstance(content, list) and len(content) == 1:
                content = content[0]
            if isinstance(content, str):
                return self._generate_text(content, system_prompt=system_prompt, **kwargs)
        except Exception as e:
            print(f"Error in generate method: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {e.args}")
            return {
                "error": type(e).__name__,
                "message": str(e),
                "details": getattr(e, 'args', None)
            }

    def _generate_text(
        self, prompt, system_prompt=None, temperature=0, max_tokens=4000, top_p=0.99, response_format=None
    ):
        if self.use_cache:
            cache_key = system_prompt + prompt if system_prompt else prompt
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none
        
        if system_prompt is None:
            messages = [
                {"role": "user", "content": prompt},
            ]
        else: 
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        response = response.choices[0].message.content


        if self.use_cache:
            self._save_cache(cache_key, response)
        return response

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)
    