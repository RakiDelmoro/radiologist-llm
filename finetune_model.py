import json
from datasets import load_dataset

def create_conversation(image, image_report):
    instruction = "Analyze this medical image and provide a detailed diagnostic report including findings, impressions"

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": image}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": image_report}
            ]
        },
    ]

    return {"messages": conversation}

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
    from unsloth import FastVisionModel
    from trl import SFTTrainer, SFTConfig
    from unsloth.trainer import UnslothVisionDataCollator

    samples = import_medical_dataset()
    model, tokenizer = FastVisionModel.from_pretrained("Qwen3-vl-2B", load_in_4bit=True, local_files_only=True, use_gradient_checkpointing = "unsloth")
    model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)

    FastVisionModel.for_training(model)
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = UnslothVisionDataCollator(model, tokenizer),
        train_dataset = samples[:30000],
        args = SFTConfig(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 30,
            # num_train_epochs = 1, # Set this instead of max_steps for full training runs
            learning_rate = 2e-4,
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.001,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none",     # For Weights and Biases

            # You MUST put the below items for vision finetuning:
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            max_length = 2048,
            dataloader_num_workers = 0,
        ),
    )

    trainer.train()
    # Save the fine-tuned model
    model.save_pretrained("fine_tuned_model")
    tokenizer.save_pretrained("fine_tuned_model")

runner()
