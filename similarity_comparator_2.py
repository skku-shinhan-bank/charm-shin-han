from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util
import torch
import gc
from transformers import (
	ElectraConfig
)
from transformers import AutoTokenizer, ElectraForSequenceClassification, AdamW
from .model.koelectra_classifier import KoElectraClassifier
from .koelectra_classification_trainer import KoElectraClassificationDataset
from .koelectra_config import KoELECTRAConfig


class SimilarityComparator:
  def __init__(self, comparator_model_path):
    electra_config = ElectraConfig.from_pretrained("monologg/koelectra-base-v3-discriminator")
    # model = AutoModel.from_pretrained(comparator_model_path).to("cuda")
    model = KoElectraClassifier.from_pretrained(pretrained_model_name_or_path = comparator_model_path, config = electra_config, num_labels = 2).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

    # no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #   {
    #     'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #     'weight_decay': 0.01
    #   },
    #   {
    #     'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    #     'weight_decay': 0.0
    #   },
    # ]
    # optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5)

    checkpoint = torch.load(comparator_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']

    self.tokenizer = tokenizer
    self.model = model
    pass

  def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

  def compare_generate_review(self, str1, data, comment):

    data.insert(0, str1)
    comment.insert(0, ".")

    #Tokenize sentences
    encoded_input = self.tokenizer(data, padding=True, truncation=True, max_length=18, return_tensors='pt')

    gc.collect()
    torch.cuda.empty_cache()

    self.model.eval()
    with torch.no_grad():
      model_output = self.model(input_ids=encoded_input["input_ids"].to("cuda"))

    attention_mask = encoded_input['attention_mask'].to("cuda")
    #Perform pooling. In this case, mean pooling
    sentence_embeddings = self.mean_pooling(model_output, attention_mask)
    cosine_scores = util.pytorch_cos_sim(sentence_embeddings, sentence_embeddings)

    org = 0
    temp = cosine_scores[org]
    temp.argsort(descending=True)

    for i in temp.argsort(descending=True)[0:10]:
      print(f"{i}. {data[org]} <> {data[i]} \nScore: {cosine_scores[org][i]:.4f}")

    for i in temp.argsort(descending=True)[1:10]:
        print("review")
        print(data[i])
        print("comment")
        print(comment[i])

