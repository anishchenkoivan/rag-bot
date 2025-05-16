import requests
import json

import config


def gemini_api_call(prompts):
    results = []
    for prompt in prompts:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {config.api_key}",
            },
            data=json.dumps({
                "model": "google/gemini-2.0-flash-001",
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })
        )
        content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        results.append(content)
    print(results)
    return results
