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

class SimilarityComparator:
  def __init__(self):
    tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    model = AutoModel.from_pretrained("monologg/koelectra-base-v3-discriminator").to("cuda")
    self.tokenizer = tokenizer
    self.model = model
    pass

  def mean_pooling(self, model_output, attention_mask):
    #Mean Pooling - Take attention mask into account for correct averaging
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

  def compare_between_two(self, str1, str2):
    dataset = []
    dataset.append(str1)
    dataset.append(str2)

    #Tokenize sentences
    encoded_input = self.tokenizer(dataset, padding=True, truncation=True, max_length=32, return_tensors='pt')

    gc.collect()
    torch.cuda.empty_cache()

    with torch.no_grad():
      model_output = self.model(input_ids=encoded_input["input_ids"].to("cuda"))

    #Perform pooling. In this case, mean pooling
    sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'].to("cuda"))
    cosine_scores = util.pytorch_cos_sim(sentence_embeddings, sentence_embeddings)

    org = 0
    temp = cosine_scores[org]
    temp.argsort(descending=True)

    for i in temp.argsort(descending=True)[1:2]:
      print(f"{dataset[org]} <> {dataset[i]} \nScore: {cosine_scores[org][i]:.4f}")

    print(cosine_scores[0][1].item())

  def compare_generate_review(self, string, config, data, label, comment, model_path):
    test_label = IssuePredictor(model_path).predict(config, string)
    data.insert(0, string)
    label.insert(0, test_label)

    #Tokenize sentences
    encoded_input = self.tokenizer(data, padding=True, truncation=True, max_length=32, return_tensors='pt')

    gc.collect()
    torch.cuda.empty_cache()

    with torch.no_grad():
      model_output = self.model(input_ids=encoded_input["input_ids"].to("cuda"))

    #Perform pooling. In this case, mean pooling
    sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'].to("cuda"))
    cosine_scores = util.pytorch_cos_sim(sentence_embeddings, sentence_embeddings)

    org = 0
    temp = cosine_scores[org]
    temp.argsort(descending=True)

    for i in temp.argsort(descending=True)[0:10]:
      print(f"{i}. {data[org]} <> {data[i]} \nScore: {cosine_scores[org][i]:.4f}")

    for i in temp.argsort(descending=True)[0:10]:
      if(label[i]==label[org]):
        print(label[i])
        print("review")
        print(data[i])
        print("comment")
        print(comment[i])


class IssuePredictor:
  def __init__(self, model_path):
    print('1. Get KoELECTRA model')
    device = torch.device("cpu")
    # model = Model
    electra_config = ElectraConfig.from_pretrained("monologg/koelectra-base-v3-discriminator")
    model = KoElectraClassifier.from_pretrained(pretrained_model_name_or_path = model_path, config = electra_config, num_labels = 7)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
      {
        'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01
      },
      {
        'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0
      },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print('2. Load KoELECTRA Classifier Model')
    model.eval()

    print('3. Get Tokenizer')
    tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    self.tokenizer = tokenizer
    self.device = device
    self.model = model
    pass

  def predict(self, config, review):
    max_len = config.max_seq_len
    batch_size = config.batch_size

    unseen_test = [[review,0]]
    # unseen_test = []
    # row = []
    # row.append(review)
    # row.append(0)
    # unseen_test.append(row)

    test_dataset = KoElectraClassificationDataset(tokenizer=self.tokenizer, device=self.device, zipped_data = unseen_test, max_seq_len = max_len)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    total_issue_info = 0
    for batch_index, data in enumerate(test_loader):
      with torch.no_grad():
        inputs = {
          'input_ids': data['input_ids'],
          'attention_mask': data['attention_mask'],
          'labels': data['labels']
        }
        outputs = self.model(**inputs)
        loss = outputs[0]
        logit = outputs[1]
        for index, real_class_id in enumerate(inputs['labels']):
          total_issue_info = logit.argmax(1)[index].item()

    return total_issue_info
		
