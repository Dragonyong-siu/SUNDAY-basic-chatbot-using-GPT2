def SUNDAY_Loss(LM_LOGITS, MC_LOGITS, lm_labels, mc_labels):
  LM_Function = nn.CrossEntropyLoss()
  MC_Function = nn.BCEWithLogitsLoss()
  LM_Loss = LM_Function(LM_LOGITS, lm_labels)
  MC_Loss = MC_Function(MC_LOGITS, mc_labels)
  Total_loss = LM_Loss + MC_Loss
  return (Total_loss, LM_Loss, MC_Loss)
