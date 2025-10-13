import os
from textSummarizer.logging import logger
from textSummarizer.entity import ModelTrainerConfig
from datasets import load_dataset, load_from_disk
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments

class ModelTrainer:
    def __init__(self, config:ModelTrainerConfig):
        self.config = config

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt)
        seq2seq_data_collector = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)

        # ✅ Convert to string and check if exists
        data_path = str(self.config.data_path)
        print(f"Looking for data at: {data_path}")
        print(f"Absolute path: {os.path.abspath(data_path)}")
        print(f"Exists: {os.path.exists(data_path)}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")
        
        dataset_samsum_pt = load_from_disk(data_path)

        #loading data
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        print(f"Original dataset size - Train: {len(dataset_samsum_pt['test'])}, Eval: {len(dataset_samsum_pt['validation'])}")
    
        train_subset = dataset_samsum_pt["test"].select(range(10))
        eval_subset = dataset_samsum_pt["validation"].select(range(10))
        
        print(f"Using subset - Train: {len(train_subset)}, Eval: {len(eval_subset)}")

        # trainer_args = TrainingArguments(
        #     output_dir = self.config.root_dir, num_train_epochs=self.config.num_train_epochs, warmup_steps= self.config.warm_steps, 
        #     per_device_train_batch_size= self.config.per_device_train_batch_size, per_device_eval_batch_size=self.config.per_device_train_batch_size,
        #     weight_decay=self.config.weight_decay, logging_steps=self.config.logging_steps, evaluation_strategy=self.config.evaluation_strategy, eval_steps= self.config.eval_steps,
        #     save_steps=1e6, gradient_accumulation_steps=self.config.gradient_accumulation_steps
        # )


        trainer_args = TrainingArguments(
        output_dir=self.config.root_dir, num_train_epochs=1, warmup_steps=5,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        weight_decay=0.01, logging_steps=1,
        logging_first_step=True,
        eval_strategy='steps', eval_steps=500, save_steps=1e6,
        gradient_accumulation_steps=16,
        disable_tqdm=False,
        report_to="none",
        dataloader_pin_memory=False
        #fp16=True, # Enable mixed precision training
        #report_to="none" # Disable wandb logging
        )

        trainer = Trainer(model=model_pegasus, args=trainer_args,
                          processing_class= tokenizer, data_collator=seq2seq_data_collector,
                          train_dataset = train_subset,
                          eval_dataset= eval_subset
                          )
        
        trainer.train()

        model_pegasus.save_pretrained(os.path.join(self.config.root_dir, "pegasus-samsum-model"))

        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))
