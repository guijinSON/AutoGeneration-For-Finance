import yaml
import os
import torch
from transformers import TrainerCallback, TrainingArguments, ProgressCallback, AutoModelForCausalLM
from datasets import load_dataset
import argparse
from trl import SFTTrainer
from src import prepare_model_for_optimized_training,count_trainable_params

# ... [all the class definitions from your code]

def load_config(yaml_file_path):
    with open(yaml_file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main(config_path):
    config = load_config(config_path)

    os.environ["WANDB_API_KEY"] = config['general']['wandb_api_key']
    os.environ["WANDB_PROJECT"] = config['general']['wandb_project']

    data_path = config['general']['data_path']
    model_name = config['general']['model_name']
    freezing_strategy = config['general']['freezing_strategy']
    base_model_path = config['general']['base_model_path']

    dataset = load_dataset(data_path, split='train')

    model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map='auto')

    model = prepare_model_for_optimized_training(
        model, 
        model_name=model_name,
        freezing_strategy=freezing_strategy,
        freeze_ratio=config['freeze_parameters']['freeze_ratio']
    )

    count_trainable_params(model)
    model.config.use_cache = False 

    training_args = TrainingArguments(
        output_dir= config['general']['ckpt'],
        report_to="wandb",
        overwrite_output_dir=True,
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        logging_steps=config['training']['logging_steps'],
        save_strategy=config['training']['save_strategy'],
        save_steps=config['training']['save_steps']
    )

    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        dataset_text_field=config['training']['dataset_text_field'],
        packing=config['training']['packing'],
        max_seq_length=config['training']['max_seq_length'],
        args=training_args
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model using configurations from a YAML file.')
    parser.add_argument('config', type=str, help='Path to the YAML configuration file.')

    args = parser.parse_args()

    main(args.config)
