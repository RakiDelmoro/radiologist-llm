import time
import torch
from PIL import Image
from datasets import load_dataset
from unsloth import FastVisionModel
from transformers import AutoProcessor

def run_model_demo():
    start = time.time()
    # image_path = "./output_image.jpg"
    # image = Image.open(image_path)

    dataset = load_dataset('itsanmolgupta/mimic-cxr-dataset')
    image = dataset['train']['image'][-4]
    findings = dataset['train']['findings'][-4]
    impression = dataset['train']['impression'][-4]
    # Save image locally for visualization
    image.save("output_image.jpg")

    model, tokenizer = FastVisionModel.from_pretrained('x-ray-agent-v2-checkpoints/checkpoint-378', local_files_only=True)

    instruction = """Describe this X-ray."""
    messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are an expert radiologist."}
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": instruction}
        ]
    }
    ]

    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens = False,
        return_tensors = "pt",
    ).to("cuda")

    with torch.inference_mode(): output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    generated_tokens = output[0][inputs['input_ids'].shape[1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(answer)
    end = time.time()
    print(f'Finish in {end - start:.2f} seconds')

    print(findings)
    print(impression)

run_model_demo()
