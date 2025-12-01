import torch
from PIL import Image
from unsloth import FastVisionModel
from transformers import AutoProcessor

def run_model_demo():
    model, tokenizer = FastVisionModel.from_pretrained("Qwen3-vl-2B", load_in_4bit=True, local_files_only=True)
    print('Finish preparing model')

    image_path = "./demo-pic.JPG"
    image = Image.open(image_path).convert('RGB')
    messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": image_path},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
    ]
    print('Finish preparing messages.')
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    processor = AutoProcessor.from_pretrained("Qwen3-vl-2B")
    inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)
    print('Finish preparing inputs for model')
    
    with torch.inference_mode(): output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    print('Finish generating answer from the model')

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    print(answer)
    print('Finish decode answer')

run_model_demo()
