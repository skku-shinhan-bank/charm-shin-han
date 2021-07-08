import pandas as pd
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, ElectraForSequenceClassification, AdamW
from .model.koelectra_classifier import koElectraForSequenceClassification
from tqdm.notebook import tqdm
import torch
import os
import matplotlib.pyplot as plt
from IPython.display import display
from tqdm import tqdm
from kobert_transformers import get_tokenizer
from transformers import (
  ElectraConfig,
)


class KoElectraClassificationEvaluate():
    def __init__(self):
        pass

    def evaluate(self, dataset_test, config, save_ckpt_path):
        ctx = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(ctx)
        eval_loss, eval_acc = self.evaluate_model(device=device, batch_size=config.batch_size, dataset_test=dataset_test, num_label=config.num_label, max_seq_len=config.max_seq_len, save_ckpt_path=save_ckpt_path)
        print(f'\tLoss: {eval_loss:.4f}(valid)\t|\tAcc: {eval_acc * 100:.1f}%(valid)')

	def get_model_and_tokenizer(self, device, save_ckpt_path, num_label):
		electra_config = ElectraConfig.from_pretrained("monologg/koelectra-small-v2-discriminator")
		model = koElectraForSequenceClassification.from_pretrained(pretrained_model_name_or_path = "monologg/koelectra-small-v2-discriminator", config = electra_config, num_labels = num_label)
		tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v2-discriminator")

		if os.path.isfile(save_ckpt_path):
			checkpoint = torch.load(save_ckpt_path, map_location=device)
			pre_epoch = checkpoint['epoch']
			model.load_state_dict(checkpoint['model_state_dict'])

			print(f"\n\nload pretrain from\n\n: {save_ckpt_path}, epoch={pre_epoch}")

		return model, tokenizer

    def get_model_input(data):
        return {'input_ids': data['input_ids'],
                    'attention_mask': data['attention_mask'],
                    'labels': data['labels']
                }

    def evaluate_model(self, device, batch_size, dataset_test, num_label, max_seq_len, save_ckpt_path):

		model, tokenizer = self.get_model_and_tokenizer(device=device, save_ckpt_path=save_ckpt_path, num_label=num_label)
		model.to(device)

		eval_dataset = self.WellnessTextClassificationDataset(device=device, tokenizer=tokenizer, zippedData=dataset_test, num_label = num_label, max_seq_len = max_seq_len)
		eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size)

		loss = 0
		acc = 0

		model.eval()
		for data in tqdm(eval_dataloader, desc="Evaluating"):
			with torch.no_grad():
				inputs = self.get_model_input(data)
				outputs = model(**inputs)
				loss += outputs[0]
				logit = outputs[1]
				acc += (logit.argmax(1)==inputs['labels']).sum().item()
				print('\n\n가나다라마바사', logit.argmax(1))

		return loss / len(eval_dataset), acc / len(eval_dataset)