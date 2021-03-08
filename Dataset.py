import torch
special_tokens = '<bos>', '<eos>', '<speaker1>', '<speaker2>', '<pad>'
tokenizer.add_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))
class SUNDAY_Dataset_T(torch.utils.data.Dataset):
  def __init__(self, data, max_len):
    self.data = data
    self.max_len = max_len
    self.pad_emb = tokenizer.convert_tokens_to_ids('<pad>')

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    Dictionary_T = {}
    Attribute_out = Arrange_attributes(self.data[index])
    FORM = Make_form(Attribute_out[0], Attribute_out[1], Attribute_out[2])
    words, segments, position, Sequence = Make_inputs(FORM[0], FORM[1], FORM[2])
    distractor = Attribute_out[3][0][0]

    Distractor = tokenizer.tokenize(distractor)
    words_distract, segments_distract, _, _ = Make_inputs(FORM[0], FORM[1], Distractor)
    
    words = tokenizer.convert_tokens_to_ids(words)
    segments = tokenizer.convert_tokens_to_ids(segments)
    words_DISTRACT = tokenizer.convert_tokens_to_ids(words_distract)
    segments_DISTRACT = tokenizer.convert_tokens_to_ids(segments_distract)
    lm_targets = words[1:]
    
    padding_length = self.max_len
    def padding(x, padding_value):
      return x + [padding_value] * (padding_length - len(x))
    
    (words, segments) = [padding(x, self.pad_emb) for x in (words[:-1], segments[:-1])]
 
    lm_targets = padding(lm_targets, self.pad_emb)
    
    input_ids_t = torch.Tensor(words).long()
    token_type_ids_t = torch.Tensor(segments).long()
    lm_labels_t = torch.Tensor(lm_targets).long()
    mc_labels_t = torch.Tensor([1]).float()

    if len(input_ids_t.tolist()) > 120:
      input_ids_t = input_ids_t[:120]
      token_type_ids_t = token_type_ids_t[:120]
      lm_labels_t = torch.Tensor(lm_labels_t.tolist()[:119] + [50258])
    
    Dictionary_T['input_ids'] = input_ids_t
    Dictionary_T['token_type_ids'] = token_type_ids_t
    Dictionary_T['lm_labels'] = lm_labels_t
    Dictionary_T['mc_labels'] = mc_labels_t

    return Dictionary_T

class SUNDAY_Dataset_F(torch.utils.data.Dataset):
  def __init__(self, data, max_len):
    self.data = data
    self.max_len = max_len
    self.pad_emb = tokenizer.convert_tokens_to_ids('<pad>')

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    Dictionary_F = {}
    Attribute_out = Arrange_attributes(self.data[index])
    FORM = Make_form(Attribute_out[0], Attribute_out[1], Attribute_out[2])
    words, segments, position, Sequence = Make_inputs(FORM[0], FORM[1], FORM[2])
    distractor = Attribute_out[3][0][0]

    Distractor = tokenizer.tokenize(distractor)
    words_distract, segments_distract, _, _ = Make_inputs(FORM[0], FORM[1], Distractor)
    
    words = tokenizer.convert_tokens_to_ids(words)
    segments = tokenizer.convert_tokens_to_ids(segments)
    words_DISTRACT = tokenizer.convert_tokens_to_ids(words_distract)
    segments_DISTRACT = tokenizer.convert_tokens_to_ids(segments_distract)
    lm_distractor = words_DISTRACT[1:]
    
    padding_length = self.max_len
    def padding(x, padding_value):
      return x + [padding_value] * (padding_length - len(x))
    
    (words_DISTRACT, segments_DISTRACT) = [padding(x, self.pad_emb) for x in (words_DISTRACT[:-1], segments_DISTRACT[:-1])]


    lm_distractor = padding(lm_distractor, self.pad_emb)
    
    input_ids_f = torch.Tensor(words_DISTRACT).long()
    token_type_ids_f = torch.Tensor(segments_DISTRACT).long()
    lm_labels_f = torch.Tensor(lm_distractor).long()
    mc_labels_f = torch.Tensor([0]).float()

    if len(input_ids_f.tolist()) > 120:
      input_ids_f = input_ids_f[:120]
      token_type_ids_f = token_type_ids_f[:120]
      lm_labels_f = torch.Tensor(lm_labels_f.tolist()[:119] + [50258])

    Dictionary_F['input_ids'] = input_ids_f
    Dictionary_F['token_type_ids'] = token_type_ids_f
    Dictionary_F['lm_labels'] = lm_labels_f
    Dictionary_F['mc_labels'] = mc_labels_f
    
    
    return Dictionary_F

class SUNDAY_Dataset_plusT(torch.utils.data.Dataset):
  def __init__(self, data, max_len):
    self.data = data
    self.max_len = max_len
    self.pad_emb = tokenizer.convert_tokens_to_ids('<pad>')

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    Dictionary_Pt = {}
    Attribute_out = Arrange_attributes(self.data[index])
    FORM = Make_form_plus(Attribute_out[0], Attribute_out[1], Attribute_out[2])
    words, segments, position, Sequence = Make_inputs(FORM[0], FORM[1], FORM[2])
    distractor = Attribute_out[3][0][0]

    Distractor = tokenizer.tokenize(distractor)
    words_distract, segments_distract, _, _ = Make_inputs(FORM[0], FORM[1], Distractor)
    
    words = tokenizer.convert_tokens_to_ids(words)
    segments = tokenizer.convert_tokens_to_ids(segments)
    words_DISTRACT = tokenizer.convert_tokens_to_ids(words_distract)
    segments_DISTRACT = tokenizer.convert_tokens_to_ids(segments_distract)
    lm_targets = words[1:]
    
    padding_length = self.max_len
    def padding(x, padding_value):
      return x + [padding_value] * (padding_length - len(x))
    
    (words, segments) = [padding(x, self.pad_emb) for x in (words[:-1], segments[:-1])]
 
    lm_targets = padding(lm_targets, self.pad_emb)
    
    input_ids_pt = torch.Tensor(words).long()
    token_type_ids_pt = torch.Tensor(segments).long()
    lm_labels_pt = torch.Tensor(lm_targets).long()
    mc_labels_pt = torch.Tensor([1]).float()

    if len(input_ids_pt.tolist()) > 120:
      input_ids_pt = input_ids_pt[:120]
      token_type_ids_pt = token_type_ids_pt[:120]
      lm_labels_pt = torch.Tensor(lm_labels_pt.tolist()[:119] + [50258])    
    
    Dictionary_Pt['input_ids'] = input_ids_pt
    Dictionary_Pt['token_type_ids'] = token_type_ids_pt
    Dictionary_Pt['lm_labels'] = lm_labels_pt
    Dictionary_Pt['mc_labels'] = mc_labels_pt

    
    return Dictionary_Pt

class SUNDAY_Dataset_plusF(torch.utils.data.Dataset):
  def __init__(self, data, max_len):
    self.data = data
    self.max_len = max_len
    self.pad_emb = tokenizer.convert_tokens_to_ids('<pad>')

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    Dictionary_Pf = {}
    Attribute_out = Arrange_attributes(self.data[index])
    FORM = Make_form_plus(Attribute_out[0], Attribute_out[1], Attribute_out[2])
    words, segments, position, Sequence = Make_inputs(FORM[0], FORM[1], FORM[2])
    distractor = Attribute_out[3][0][0]

    Distractor = tokenizer.tokenize(distractor)
    words_distract, segments_distract, _, _ = Make_inputs(FORM[0], FORM[1], Distractor)
    
    words = tokenizer.convert_tokens_to_ids(words)
    segments = tokenizer.convert_tokens_to_ids(segments)
    words_DISTRACT = tokenizer.convert_tokens_to_ids(words_distract)
    segments_DISTRACT = tokenizer.convert_tokens_to_ids(segments_distract)
    lm_distractor = words_DISTRACT[1:]
    
    padding_length = self.max_len
    def padding(x, padding_value):
      return x + [padding_value] * (padding_length - len(x))
    
    (words_DISTRACT, segments_DISTRACT) = [padding(x, self.pad_emb) for x in (words_DISTRACT[:-1], segments_DISTRACT[:-1])]
 
    lm_distractor = padding(lm_distractor, self.pad_emb)
    
    input_ids_pf = torch.Tensor(words_DISTRACT).long()
    token_type_ids_pf = torch.Tensor(segments_DISTRACT).long()
    lm_labels_pf = torch.Tensor(lm_distractor).long()
    mc_labels_pf = torch.Tensor([0]).float()

    if len(input_ids_pf.tolist()) > 120:
      input_ids_pf = input_ids_pf[:120]
      token_type_ids_pf = token_type_ids_pf[:120]
      lm_labels_pf = torch.Tensor(lm_labels_pf.tolist()[:119] + [50258])    
    
    Dictionary_Pf['input_ids'] = input_ids_pf
    Dictionary_Pf['token_type_ids'] = token_type_ids_pf
    Dictionary_Pf['lm_labels'] = lm_labels_pf
    Dictionary_Pf['mc_labels'] = mc_labels_pf

    
    return Dictionary_Pf

def Arrange_attributes(row):
  Personality =row['personality']
  Utterances = row['utterances']
  Candidates = []
  Next_sententences = []
  History = []
  for i in range(len(Utterances)):
    candidates = Utterances[i]['candidates']
    next_sentence = [Utterances[i]['candidates'][-1]]
    history = Utterances[i]['history']
    Candidates.append(candidates)
    Next_sententences.append(next_sentence)
    History.append(history)

  return (Personality, History, Next_sententences, Candidates, Utterances)


def Make_form(personas, History, Next_sententences):
  PERSONALITY = []
  for i in range(len(personas)):
    personas_tokens = tokenizer.tokenize(personas[i])
    PERSONALITY.append(personas_tokens)
    
  HISTORY = []
  for i in range(len(History[1])):
    History_tokens = tokenizer.tokenize(History[1][i])
    HISTORY.append(History_tokens)

  answer = "".join(Next_sententences[1])
  ANSWER = tokenizer.tokenize(answer)

  return (PERSONALITY, HISTORY, ANSWER)

def Make_form_plus(personas, History, Next_sentences):
  PERSONALITY = []
  for i in range(len(personas)):
    personas_tokens = tokenizer.tokenize(personas[i])
    PERSONALITY.append(personas_tokens)

  HISTORY = []
  for i in range(len(History[-1][-3:])):
    History_tokens = tokenizer.tokenize(History[-1][-3:][i])
    HISTORY.append(History_tokens)

  answer = "".join(Next_sentences[-1])
  ANSWER = tokenizer.tokenize(answer)

  return (PERSONALITY, HISTORY, ANSWER)
  
def Make_inputs(Persona, conversation, answer):
  personality = [y for x in Persona for y in x]
  history = conversation
  reply = answer
  bos, eos, speaker1, speaker2 = '<bos>', '<eos>', '<speaker1>', '<speaker2>'
  sequence = [[bos] + personality] + history + [reply + [eos]]
  Sequence = [sequence[0]] + [[speaker2 if (len(sequence) - i) % 2 else speaker1] 
                              + s for i, s in enumerate(sequence[1:])]
    
  words = [y for x in Sequence for y in x]
  segments = [speaker2 if i % 2 else speaker1
              for i, s in enumerate(Sequence) for word in s]
  position = list(range(len(words)))

  return words, segments, position, Sequence

from torch.utils.data import DataLoader
SUNDAY_Dataset = SUNDAY_Dataset_T(dataset['train'], 120) + SUNDAY_Dataset_F(dataset['train'], 120) + SUNDAY_Dataset_plusT(dataset['train'], 120) + SUNDAY_Dataset_plusF(dataset['train'], 120)
BATCH_SIZE = 16
train_dataloader = DataLoader(SUNDAY_Dataset, batch_size = BATCH_SIZE, shuffle = True, drop_last = True)
