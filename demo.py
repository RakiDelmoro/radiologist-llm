import torch
from PIL import Image
from transformers import AutoProcessor
from unsloth import FastVisionModel

def run_model_demo():
    model, tokenizer = FastVisionModel.from_pretrained(
    model_name="qwen3vl-2b",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth")
    FastVisionModel.for_inference(model)

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

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    processor = AutoProcessor.from_pretrained("qwen3vl-2b")
    inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)
    with torch.inference_mode(): output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    print(answer)

run_model_demo()
