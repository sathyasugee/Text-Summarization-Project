import os
from textSummarizer.logging import logger
from datasets import load_dataset, load_from_disk,load_metric
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments
import pandas as pd

class ModelEvalution:
    def __init__(self, config:ModelEvalutionConfig):
        self.config = config

    def generate_batch_sized_chunks(list_of_elements, batch_size):
        """split the dataset into smaller batches that we can process simultaneously
        Yield successive batch-sized chunks from list_of_elements."""

        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]

    def calculate_metric_on_test_ds(dataset, metric, model, tokenizer,
                                        batch_size=16, device=device,
                                        column_text="article",
                                        column_summary="highlights"):
        article_batches = list(generate_batch_sized_chunks(dataset[column_text], batch_size))
        target_batches = list(generate_batch_sized_chunks(dataset[column_summary], batch_size))

        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches), total=len(article_batches)):

            inputs = tokenizer(article_batch, max_length=1024,  truncation=True,
                        padding="max_length", return_tensors="pt")

            summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                            attention_mask=inputs["attention_mask"].to(device),
                            length_penalty=0.8, num_beams=8, max_length=128)
            """paramter"""

            decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=True)
                for s in summaries]
            metric.add_batch(predictions=decoded_summaries, references=target_batch)

        #  Finally compute and return the ROUGE scores.
        score = metric.compute()
        return score
    
    def evalute(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)

        dataset_samsum_pt = load_from_disk(self.config.model_path)

        rouge_names = ("rouge1", "rouge2", "rougeL", "rougeLsum")

        rouge_metric = load_metric('rouge')

        score = self.calculate_metric_on_test_ds(dataset_samsum_pt['test'][0:10], rouge_metric, model_pegasus, tokenizer, batch_size = 2, column_text = 'dialogue', column_summary= 'summary')

        row_dict = dict((rn, score[rn]) for rn in rouge_names )

        df=pd.DataFrame([row_dict], columns = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'])

        df.to_csv (self.config.metric_file_name, index= False)