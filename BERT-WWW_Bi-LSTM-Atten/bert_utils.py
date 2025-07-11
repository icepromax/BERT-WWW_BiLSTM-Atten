import os
from transformers import BertTokenizer
import torch
from bert_get_data_model import BertClassifier,BertClassifierWithAttention

bert_name='./RoBERTa_wwm_ext'
toknizer=BertTokenizer.from_pretrained(bert_name)
device='cuda:0' if torch.cuda.is_available() else 'cpu'

save_models='./bert_checkpoint_waimai'
model=BertClassifierWithAttention()
model.load_state_dict(torch.load(os.path.join(save_models,'best_atten.pt')))
model=model.to(device)
model.eval()

real_labels=['消极','积极']

while True:
    text=input("请输入酒店评论:")
    if text=='exit':
        break
    bert_input=toknizer(
        text,
        padding='max_length',
        max_length=80,
        truncation=True,#超过模型最大长度的输入会被截断
        return_tensors='pt'#自动返回维度[1,35]
    )
    input_ids=bert_input['input_ids'].to(device)
    mask=bert_input['attention_mask'].unsqueeze(1).to(device)
    print(input_ids.size())
    print(mask.size())
    output=model(input_ids,mask)
    pred=output.argmax(dim=1)
    print(real_labels[pred])
