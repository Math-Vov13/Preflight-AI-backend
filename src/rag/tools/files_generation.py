from os import environ as env

import requests
from dotenv import load_dotenv
from llama_index.core.tools import FunctionTool

load_dotenv()


def generate_image(prompt: str) -> str:
    """Generate an image from a text prompt via SiliconFlow (FLUX.2-flex).

    The prompt should be in English. Returns a URL to the generated image,
    or a short error string on failure. Render images in markdown as
    ``![title](url)``.
    """
    api_key = env.get("SILICONFLOW_API_KEY")
    if not api_key:
        return "Error generating image: SILICONFLOW_API_KEY not set."

    response = requests.post(
        "https://api.siliconflow.com/v1/images/generations",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json={
            "prompt": prompt,
            "model": "black-forest-labs/FLUX.2-flex",
            "image_size": "512x512",
        },
        timeout=120,
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        return f"Error generating image: HTTP {response.status_code} — {exc}"

    payload = response.json()
    images = payload.get("images") or []
    if not images:
        return f"Error generating image: {payload.get('message', 'no image returned')}"
    first = images[0]
    return first if isinstance(first, str) else first.get("url", "Error: no image url")


generate_image_tool = FunctionTool.from_defaults(
    fn=generate_image,
    name="generate_image",
)
