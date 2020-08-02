from tqdm import tqdm
def TRAIN_DEF(dataloader, model, optimizer, device):
  model.train()
  BOOK = tqdm(dataloader, total = len(dataloader))
  total_loss = 0.0
  
  for bi, Dictionary in enumerate(BOOK):
    input_ids = Dictionary ['input_ids']
    token_type_ids = Dictionary ['token_type_ids']
    lm_labels = Dictionary ['lm_labels']
    mc_labels = Dictionary ['mc_labels']
    input_ids = input_ids.to(device).long()
    token_type_ids = token_type_ids.to(device).long()
    lm_labels = lm_labels.to(device).long()
    mc_labels = mc_labels.to(device)
    
    optimizer.zero_grad()
    LM_LOGITS, MC_LOGITS = model(input_ids, token_type_ids = token_type_ids)
    LM_LOGITS = LM_LOGITS.view(-1, 50262, 400)
    lm_labels = lm_labels.view(-1, 400)
    
    LOSS = SUNDAY_Loss(LM_LOGITS, MC_LOGITS, lm_labels, mc_labels)
    LM_Loss = LOSS[1].view(-1) * 0.6
    MC_Loss = LOSS[2].view(-1) * 2
    
    del LM_LOGITS
    del MC_LOGITS
    del lm_labels
    del mc_labels
    del input_ids
    del token_type_ids

    Loss = LOSS[0]
    LM_Loss.backward(retain_graph =True)
    MC_Loss.backward()
    optimizer.step()
    total_loss += Loss.item()
  average_train_loss = total_loss / len(dataloader)
  print(" Average training loss: {0:.2f}".format(average_train_loss))  

lr = 2e-5
def FIT(dataloader, EPOCHS = 15):
  optimizer = torch.optim.AdamW(SUNDAY.parameters(), lr = lr)
   
  for i in range(EPOCHS):
    print(f"EPOCHS:{i+1}")
    print('TRAIN')
    TRAIN_DEF(train_dataloader, SUNDAY, optimizer, device)    
    torch.save(SUNDAY, '/content/gdrive/My Drive/' + f'SUNDAY_model:{i+1}')
    
FIT(train_dataloader)
