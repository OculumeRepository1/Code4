import json
import base64
from PIL import Image
from io import BytesIO

def encode_image_to_base64(image_path):
    """Encodes a PIL image as base64 string."""
    image = Image.open(image_path)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64

def save_all_images_with_prompts_and_captions_to_json(image_paths, captions, json_path, prompt=None):
    if len(image_paths) != len(captions):
        raise ValueError("The number of image paths and captions must match.")

    if prompt is None:
        prompt = "Does this image contain any signs of violence, weapons, smoking, or someone fallen?"

    all_data = []

    for image_path, caption in zip(image_paths, captions):
        #mg_base64 = encode_image_to_base64(image_path)
        entry = {
            "image_path": image_path,
            #"prompt": prompt,    # ✅ Added
            "caption": caption,  # assistant response
            #"image": [img_base64]  # base64
        }
        all_data.append(entry)

    with open(json_path, 'w') as f:
        json.dump(all_data, f)

    print(f"All images, prompts, and captions saved to {json_path}")

# Example usage
captions = [
    """No.\n\n**Reasoning:**\n\nThe image shows some people sitting in a room using phones, listening to music. One person is standing and there is no violence or weapon visible. Also, there are no drugs in the scene.""",
    """No.\n\n**Reasoning:**\n\nThe image shows a kid listening to music. There is no violence or weapon visible. Also, there are no drugs in the scene.""",
    """No.\n\n**Reasoning:**\n\nThe image shows a person standing at a front desk. Nothing suspicious about it, and there is no violence or weapon visible. Also, there are no drugs in the scene.""",
    """No.\n\n**Reasoning:**\n\nThe image shows railings and a cycle stand. There is nothing suspicious in the scene and there is no violence or weapon visible. Also, there are no drugs in the scene."""
]

image_paths = ["Data/alert.jpg", "Data/alert(1).jpg", "Data/alert(2).jpg", "Data/alert(3).jpg"]

# Save everything in one JSON file
save_all_images_with_prompts_and_captions_to_json(
    image_paths,
    captions,
    "all_image_captions.json"
)