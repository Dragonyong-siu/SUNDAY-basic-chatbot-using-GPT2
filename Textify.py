import random
def NEXT_word_index(Logits):
  Last_word_embedding = Logits[0, -1]
  Words_probability = torch.softmax(Last_word_embedding, dim = 0)
  Words_probability = Words_probability.tolist()
  Words_Sorted = sorted(Words_probability)

  First_value = Words_Sorted[-1]
  Second_value = Words_Sorted[-2]
  Third_value = Words_Sorted[-3]
  Fourth_value = Words_Sorted[-4]
  Fifth_value = Words_Sorted[-5]
  
  First_index = Words_probability.index(First_value)
  Second_index = Words_probability.index(Second_value)
  Third_index = Words_probability.index(Third_value)
  Fourth_index = Words_probability.index(Fourth_value)
  Fifth_index = Words_probability.index(Fifth_value)

  index_LIST = [First_index, Second_index, Third_index, Fourth_index, Fifth_index]
  INDEX_LIST = []
  for index in index_LIST:
    if index >= 0.15:
      INDEX_LIST.append(index)

  Words_index = random.choice(INDEX_LIST)

  return Words_index

def Make_talk(persona, history):
  input = []
  Bos_embedding = tokenizer.encode('<bos>')
  Speaker1_embedding = tokenizer.encode('<speaker1>')
  for i in range(len(persona)):
    input.append(persona[i])
  
  for j in range(len(history)):
    Talker = ('<speaker1>' if j % 2 else '<speaker2>')
    input.append(Talker + history[j])
  
  encoding = []
  for k in range(len(input)):
    input_encoded = tokenizer.encode(input[k])
    encoding.append(input_encoded)

  input_ids = [Bos_embedding] + encoding + [Speaker1_embedding]
  input_ids = [y for x in input_ids for y in x]
  token_types = ['<speaker2>' if i % 2 else '<speaker1>' 
                 for i, s in enumerate(encoding) for word in s]

  token_type_ids = Bos_embedding + tokenizer.encode(token_types) + Speaker1_embedding
  return input_ids, token_type_ids

def Make_sample(input_ids, token_type_ids, tokenizer, model):
  model.eval()
  original_length = len(input_ids.squeeze(0))
  with torch.no_grad():
     Pad_embedding = tokenizer.encode('<pad>')
     Speaker1_embedding = tokenizer.encode('<speaker1>')
     Speaker2_embedding = tokenizer.encode('<speaker2>')
     pad = []
     speaker1 = []
     speaker2 = []
     next_words = []
     while len(pad) <= 2 and len(speaker1) < 1 and len(speaker2) < 1 and len(next_words) <10:
       Logits = model(input_ids, token_type_ids = token_type_ids)
       LM_LOGITS = Logits[0]
       next_token_ids = NEXT_word_index(LM_LOGITS)
       next_words.append(next_token_ids)
       input_ids = torch.cat([input_ids, torch.Tensor([[next_token_ids]]).to(device).long()], dim = 1)
       if next_token_ids == Pad_embedding:
         pad.append(next_token_ids)
         Changed_token_type_ids = token_type_ids.tolist() + [Pad_embedding]
         Changed_token_type_ids = [y for x in Changed_token_type_ids for y in x]
         token_type_ids = torch.Tensor([Changed_token_type_ids]).to(device).long()
       
       elif next_token_ids == Speaker1_embedding:
         speaker1.append(Speaker1_embedding)
         Changed_token_type_ids = token_type_ids.tolist() + [Speaker1_embedding]
         Changed_token_type_ids = [y for x in Changed_token_type_ids for y in x]
         token_type_ids = torch.Tensor([Changed_token_type_ids]).to(device).long()

       elif next_token_ids == Speaker2_embedding:
         speaker1.append(Speaker2_embedding)
         Changed_token_type_ids = token_type_ids.tolist() + [Speaker2_embedding]
         Changed_token_type_ids = [y for x in Changed_token_type_ids for y in x]
         token_type_ids = torch.Tensor([Changed_token_type_ids]).to(device).long()
       
       else:
         Changed_token_type_ids = token_type_ids.tolist() + [Speaker2_embedding]
         Changed_token_type_ids = [y for x in Changed_token_type_ids for y in x]
         token_type_ids = torch.Tensor([Changed_token_type_ids]).to(device).long()
  
  changed_length = len(input_ids.squeeze(0))
  Sample_length = changed_length - original_length
  return input_ids, token_type_ids, Sample_length


def SUNDAY_Converation(Personality, model):
  History = []
  model.eval()
  while True:
    Chat_text = input(">>> ")
    while not Chat_text:
      print('You should say something to SUNDAY, sir')
      Chat_text = input(">>> ")
    History.append(Chat_text)
    with torch.no_grad():
      input_ids, token_type_ids = Make_talk(Personality, History)
      input_ids = torch.Tensor(input_ids).long().unsqueeze(0).to(device)
      token_type_ids = torch.Tensor(token_type_ids).long().unsqueeze(0).to(device)
 
      sample_ids1, sample_type_ids1, Sample_length1 = Make_sample(input_ids, token_type_ids, tokenizer, model)
      sample_ids2, sample_type_ids2, Sample_length2 = Make_sample(input_ids, token_type_ids, tokenizer, model)

      _, MC_Logits1 = model(sample_ids1, sample_type_ids1)
      _, MC_Logits2 = model(sample_ids2, sample_type_ids2)

      mc_checker = nn.Sigmoid()
      MC1 = mc_checker(MC_Logits1)
      MC2 = mc_checker(MC_Logits2)
      if MC1 > MC2:
        sample_ids = sample_ids1
        Sample_length = Sample_length1
      else:
        sample_ids = sample_ids2
        Sample_length = Sample_length2

      Sample = tokenizer.decode(sample_ids.squeeze(0).tolist())
  
    Conversation = tokenizer.decode(sample_ids.squeeze(0)[- Sample_length :])
    History.append(Conversation)
    print(Conversation)
