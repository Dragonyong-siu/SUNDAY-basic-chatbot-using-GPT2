#0. Download and load 'data'
import json
from pytorch_pretrained_bert import cached_path

url = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"
dataset_File = cached_path(url)
with open(dataset_File, mode = "r", encoding = "utf-8") as file:
  dataset = json.loads(file.read())
  
#0. Analyze 'dataset'
print('dataset의 data 개수 :', len(dataset['train']))
print('data의 정보 :', 'personality /', 'utterances = {candidates, history}')

from transformers import GPT2Tokenizer, GPT2Model, GPT2Config
config = GPT2Config.from_pretrained('gpt2', output_hidden_states=True)
model = GPT2Model(config).from_pretrained('gpt2', config = config)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
