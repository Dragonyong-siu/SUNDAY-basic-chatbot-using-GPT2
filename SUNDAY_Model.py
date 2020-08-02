device = 'cuda'
import torch.nn as nn
import numpy as np
#2. SUNDAY_Model
class SUNDAY_Model(nn.Module):
  def __init__(self):
    super(SUNDAY_Model, self).__init__()
    self.GPT2_Model = model
    self.dropout = nn.Dropout(0.3)
    self.linear_LM = nn.Linear(768, 50262)
    self.linear_MC = nn.Linear(768 * 3, 1)
    self.activation = nn.Sigmoid()
    nn.init.normal_(self.linear_LM.weight, std=0.02)
    nn.init.normal_(self.linear_LM.bias, 0)
    nn.init.normal_(self.linear_MC.weight, std=0.02)
    nn.init.normal_(self.linear_MC.bias, 0)

  def forward(self, input_ids, token_type_ids):
    Last_hidden, _, _ = self.GPT2_Model(input_ids, token_type_ids = token_type_ids)
    LM_LOGITS = self.dropout(Last_hidden)
    LM_LOGITS = self.linear_LM(LM_LOGITS)
    Apool = torch.mean(Last_hidden, dim = 1)
    Mpool, _ = torch.max(Last_hidden, dim = 1)
    Ppool = 0.2 * Apool + 0.8 * Mpool
    Concat = torch.cat((Apool, Mpool, Ppool), dim = 1)
    MC_LOGITS = self.dropout(Concat)
    MC_LOGITS = self.linear_MC(MC_LOGITS)
    MC_LOGITS = self.activation(MC_LOGITS)

    return (LM_LOGITS, MC_LOGITS) 
SUNDAY = SUNDAY_Model().to(device)
