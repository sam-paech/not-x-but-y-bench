# src/api.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
import threading
from typing import Optional, Dict, Any

import requests


class ApiClient:
    """
    OpenAI-compatible and Anthropic client using requests.
    Detection is based on base_url.
    """

    def __init__(self, base_url: str, api_key: str, timeout: float = 120.0, max_retries: int = 5, retry_delay: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.kind = self._detect_kind(self.base_url)
        self._local = threading.local()

    @staticmethod
    def _detect_kind(base_url: str) -> str:
        u = base_url.lower()
        if "anthropic" in u:
            return "anthropic"
        if "openrouter" in u:
            return "openrouter"
        # default to openai-compatible
        return "openai"

    def _session(self) -> requests.Session:
        sess = getattr(self._local, "session", None)
        if sess is None:
            sess = requests.Session()
            self._local.session = sess
        return sess

    def _request(self, method: str, url: str, headers: Dict[str, str], payload: Dict[str, Any], retries: int = 5) -> Dict[str, Any]:
        backoff = 1.0
        last_error: Optional[Exception] = None
        for attempt in range(retries):
            try:
                resp = self._session().request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if resp.status_code >= 200 and resp.status_code < 300:
                    return resp.json()
                # Retry on 429 and 5xx
                if resp.status_code in (429,) or 500 <= resp.status_code < 600:
                    time.sleep(backoff)
                    backoff = min(30.0, backoff * 2.0)
                    continue
                # Non-retryable
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
            except requests.RequestException as e:
                last_error = e
                time.sleep(backoff)
                backoff = min(30.0, backoff * 2.0)
        raise RuntimeError(f"Request failed after {retries} retries: {last_error}")

    def generate(self, model: str, prompt_text: str, max_tokens: int = 2048) -> str:
        """
        Always temperature=0.7.
        Use min_p=0.1 when supported (OpenRouter).
        Retries on failure with configurable retry count and delay.
        """
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                if self.kind == "anthropic":
                    return self._anthropic_generate(model, prompt_text, max_tokens=max_tokens)
                return self._openai_compat_generate(model, prompt_text, max_tokens=max_tokens)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
        raise RuntimeError(f"Generate failed after {self.max_retries} retries: {last_error}")

    def _openai_compat_generate(self, model: str, prompt_text: str, max_tokens: int) -> str:
        url = self.base_url
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Write 1000 words on the provided writing prompt."},
                {"role": "user", "content": prompt_text},
            ],
            "temperature": 0.7,
            "max_tokens": max_tokens,
        }
        # OpenRouter supports min_p
        if self.kind == "openrouter":
            body["min_p"] = 0.1

        data = self._request("POST", url, headers, body)
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            raise RuntimeError(f"Bad response: {json.dumps(data)[:1000]}")

    def _anthropic_generate(self, model: str, prompt_text: str, max_tokens: int) -> str:
        # Anthropic Messages API'
        url = self.base_url
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "Write 1000 words on the provided writing prompt."}]},
                {"role": "user", "content": [{"type": "text", "text": prompt_text}]}
            ],
        }
        data = self._request("POST", url, headers, body)
        try:
            parts = data.get("content", [])
            texts = [p.get("text", "") for p in parts if p.get("type") == "text"]
            return "\n".join(texts).strip()
        except Exception:
            raise RuntimeError(f"Bad response: {json.dumps(data)[:1000]}")
