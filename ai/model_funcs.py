import os
import logging
import asyncio
import aiofiles
from asyncio import Semaphore
from typing import Dict, Any, Optional
import weakref
import time
from dataclasses import dataclass
from functools import lru_cache

from funcs import encode_img



async def create_response(model,
                          prompt: str,
                          text: str,
                          client,
                          temperature: float = 0.7,
                          top_p: float = 0.9,
                          fp: float = 0.2,
                          presence_penalty: float = 0.2,
                          max_tokens: int | None=None,
                          img_path: str | None=None,
                          provider: str | None=None,
                          headers: dict | None=None,
                          timeout: float = 30.0):
    messages = await _prepare_messages(prompt, text, img_path)

    params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": fp,
        "presence_penalty": presence_penalty,
        "max_tokens": max_tokens,
    }
    
    loop = asyncio.get_event_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7
                )
            ),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        logging.warning(f"Request timeout after {timeout} seconds")
        raise






@lru_cache(maxsize=100)
async def _prepare_messages(prompt: str, text: str, img_path: str | None):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": []}
    ]

    if img_path:
        await _validate_image_path(img_path)
        img_b64 = await encode_img(img_path)
        messages[1]["content"].extend([
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
        ])
    else:
        messages[1]["content"] = text

    return messages



async def _validate_image_path(img_path: str):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Img Path NOT Found: {img_path}")

    if not os.access(img_path, os.R_OK):
        raise PermissionError(f"Нет прав на чтение изображения: {img_path}")
    
    try:
        async with aiofiles.open(img_path, 'rb') as f:
            await f.read(1)
    except Exception as e:
        raise PermissionError(f"Ошибка доступа к файлу: {str(e)}")
