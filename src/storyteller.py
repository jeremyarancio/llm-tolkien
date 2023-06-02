from typing import Dict
import requests
import json
import os


API_TOKEN = os.getenv("HUGGINGFACE_INFERENCE_TOKEN")
API_URL = "https://lbkepqwvqtxvvdfp.eu-west-1.aws.endpoints.huggingface.cloud"


class StoryTeller():
    def __init__(self):
        self.headers = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}
        

    def __call__(self, payload: Dict) -> str:
        """
        Args:
            payload (Dict): Data containing the initial text and generation parameters.
        Returns:
            str: Story
        """
        headers = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}
        data = json.dumps(payload)
        response = requests.request("POST", API_URL, headers=self.headers, data=data)
        return json.loads(response.content.decode("utf-8"))


if __name__ == "__main__":
    # Test
    storyteller = StoryTeller()
    payload = {
        "inputs": "Aragorn pulled out his sword from his shealth and",
        "parameters": {
            "max_new_tokens": 100,
            "do_sample": False,
            "temperature": 0.5,
            "repetition_penalty": 2.0,
            "forced_eos_token_id": 2
        }
    }
    story = storyteller(payload)
    print(story)