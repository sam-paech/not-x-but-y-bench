# src/config.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Tuple

def load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass  # optional dependency

def get_api_config() -> Tuple[str, str]:
    base_url = os.getenv("BASE_URL") or os.getenv("base_url")
    api_key = os.getenv("API_KEY") or os.getenv("api_key")
    if not base_url or not api_key:
        raise RuntimeError("BASE_URL and API_KEY must be set in environment or .env")
    return base_url.rstrip("/"), api_key
