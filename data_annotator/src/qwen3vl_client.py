# src/qwen3vl_client.py
from openai import OpenAI
import base64
from PIL import Image
import io
from typing import List, Optional

class Qwen3VLClient:
    def __init__(self, api_base: str, model_name: str):
        self.client = OpenAI(base_url=api_base, api_key="EMPTY")
        self.model_name = model_name

    def _encode_image(self, image_path: str) -> str:
        with Image.open(image_path) as img:
            if img.format not in ['JPEG', 'JPG']:
                img = img.convert('RGB')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            return base64.b64encode(img_bytes.getvalue()).decode("utf-8")

    def annotate_clip(
        self, 
        prompt: str, 
        image_paths: Optional[List[str]] = None
    ) -> str:
        """
        Send a prompt with optional images to the Qwen-VL model.
        
        Args:
            prompt (str): The text prompt.
            image_paths (Optional[List[str]]): List of image file paths. If None or empty, only text is sent.
        
        Returns:
            str: Model response.
        """
        # Normalize image_paths
        if image_paths is None:
            image_paths = []

        messages = [
            {
                "role": "system",
                "content": "You are an expert in robotic task analysis."
            },
            {
                "role": "user",
                "content": []
            }
        ]
        
        user_content = [{"type": "text", "text": prompt}]
        
        # Only add images if provided
        for img_path in image_paths:
            b64_img = self._encode_image(img_path)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
            })
        
        messages[1]["content"] = user_content

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=1024,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()