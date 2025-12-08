from datasets import load_dataset
from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator

def create_conversation(image, report):
    conversation = {
    "messages": [
        {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are an expert radiologist."}
        ]
        },
        {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this X-ray."},
            {"type": "image", "image": image} 
        ]
        },
        {
        "role": "assistant",
        "content": [
            {"type": "text", "text": report}
        ]
        }
    ]
    }

    return conversation

def import_medical_dataset():
    dataset = load_dataset('itsanmolgupta/mimic-cxr-dataset')

    processed_samples = []
    for sample in dataset['train']:
        report_parts = []
        
        if 'findings' in sample and sample['findings']:
            report_parts.append(f"FINDINGS:\n{sample['findings']}")

        if 'impression' in sample and sample['impression']:
            report_parts.append(f"\nIMPRESSION:\n{sample['impression']}")

        full_report = "\n".join(report_parts) if report_parts else sample.get('findings', '')
        
        processed_sample = create_conversation(sample['image'], full_report)
        processed_samples.append(processed_sample)
    
    return processed_samples

def runner():
    samples = import_medical_dataset()
    model, tokenizer = FastVisionModel.from_pretrained("./base-2b", local_files_only=True, load_in_4bit=True, use_gradient_checkpointing=False)
    model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers
    r = 32,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 32 # Recommended alpha == r at least
)
    FastVisionModel.for_training(model)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=samples[:30000],
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=16,
            warmup_steps=5,
            # max_steps=30,
            num_train_epochs=1,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            remove_unused_columns=False,
            dataset_text_field="messages",  # REQUIRED
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=2048,
            output_dir="x-ray-agent-v2-checkpoints",
            report_to="none",
            save_steps=1,
            save_strategy="steps",
            save_total_limit=5,
        ),
    )

    trainer.train(resume_from_checkpoint='x-ray-agent-v2-checkpoints/checkpoint-236')
    # Save the fine-tuned model
    model.save_pretrained("x-ray-agent-v2")
    tokenizer.save_pretrained("x-ray-agent-v2")

runner()
