import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer,RobertaTokenizer
from torch import nn
from transformers import BertModel,RobertaModel
from sklearn.model_selection import train_test_split


bert_name='./RoBERTa_wwm_ext'
tokenizer=BertTokenizer.from_pretrained(bert_name)
class MyDataset(Dataset):
    def __init__(self,df):
        self.texts=[
            tokenizer(
                text,
                padding='max_length',
                max_length=80,
                truncation=True,#自动截断
                return_tensors="pt"
            ) for text in df['text']
        ]
        self.labels=[label for label in df['label']]

    def __getitem__(self, idx):
        return self.texts[idx],self.labels[idx]
    def __len__(self):
        return len(self.labels)

class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier,self).__init__()
        self.bert=BertModel.from_pretrained(bert_name)
        self.dropout=nn.Dropout(0.5)
        self.dropout2=nn.Dropout(0.4)
        self.linear=nn.Linear(768,100)
        self.linear1=nn.Linear(100,2)
        self.relu=nn.ReLU()

    def forward(self,input_id,mask):
        _,pooler_output=self.bert(input_ids=input_id,attention_mask=mask,return_dict=False)
        dropout_output=self.dropout(pooler_output)

        linear_output=self.dropout2(self.linear(dropout_output))

        linear_output1=self.linear1(linear_output)

        final_layer=self.relu(linear_output1)

        return final_layer


class ParametricAttention(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        # 参数向量（将矩阵F改为向量形式）
        self.w = nn.Parameter(torch.Tensor(hidden_dim))
        self.b = nn.Parameter(torch.zeros(1))  # 标量偏置

        # 初始化
        nn.init.normal_(self.w, mean=0, std=0.02)  # 小随机初始化
        self.scale = hidden_dim ** -0.5

    def forward(self, hidden_states, mask=None):
        """
        参数:
            hidden_states: [batch_size, seq_len, hidden_dim]
            mask: [batch_size, 1,seq_len]
        返回:
            context: [batch_size, hidden_dim]  # 二维
            attn_weights: [batch_size, seq_len]  # 二维
        """
        # 式1: u_t = tanh(w^T h_t + b) → 输出标量

        mask=mask.squeeze(1)
        # print("mask", mask.shape)
        scores = torch.tanh(
            torch.matmul(hidden_states, self.w) + self.b  # [batch, seq_len]
        ) * self.scale  # 缩放保持数值稳定
        # 处理mask

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # 式2: α_t = softmax(u_t)
        attn_weights = torch.softmax(scores, dim=-1)  # [batch, seq_len]

        # 式3: c = Σ(α_t * h_t)
        context = torch.sum(
            attn_weights.unsqueeze(-1) * hidden_states,  # [batch, seq_len, 1] * [batch, seq_len, hidden_dim]
            dim=1
        )  # [batch, hidden_dim]


        return context, attn_weights

class BertClassifierWithAttention(nn.Module):
    def __init__(self):
        super(BertClassifierWithAttention,self).__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.bilstm = nn.LSTM(
            input_size=768,
            hidden_size=384,
            bidirectional=True,
            batch_first=True
        )
        self.attention = ParametricAttention(hidden_dim=768)  # BiLSTM输出维度=384 * 2
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, 100),
            nn.Dropout(0.4),
            nn.Linear(100, 2),
            nn.ReLU()
        )

    def forward(self, input_ids, attention_mask):
        # 获取BERT所有token的隐藏状态（而非pooler_output）
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state  # [batch, seq_len, 768]

        # BiLSTM处理序列
        lstm_out, _ = self.bilstm(sequence_output)  # [batch, seq_len, 768]

        # 自注意力
        attn_out,_ = self.attention(lstm_out,mask=attention_mask)  # [batch, seq_len, 768]

        # 分类
        logits = self.classifier(attn_out)

        return logits

class BertClassifier1(nn.Module):
    def __init__(self):
        super(BertClassifier1,self).__init__()
        self.bert=BertModel.from_pretrained(bert_name)
        self.bilstm = nn.LSTM(
            input_size=768,  # BERT隐藏层维度
            hidden_size=384,
            bidirectional=True,
            batch_first=True,
            num_layers=1
        )
        self.linear=nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768,100),
            nn.Dropout(0.4),
            nn.Linear(100,2),
            nn.ReLU()
        )

    def forward(self,input_id,mask):
        batch=input_id.size(0)
        _,pooler_output=self.bert(input_ids=input_id,attention_mask=mask,return_dict=False)
        pooler_output,_=self.bilstm(pooler_output)#[batch,input_size]

        final_layer=self.linear(pooler_output)
        return final_layer

def GenerateDate(mode='train'):
        data_path= './data/cleaned_ChnSentiCorp_htl_all.csv'
        data=pd.read_csv(data_path)
        data['text'] = data['text'].astype(str)
        train_data,test_data=train_test_split(data,test_size=0.2,shuffle=True,random_state=42)
        train_data,val_data=train_test_split(train_data,test_size=0.1,shuffle=True,random_state=42)

        train_dataset=MyDataset(train_data)
        test_dataset=MyDataset(test_data)
        val_dataset=MyDataset(val_data)

        if mode=='train':
            return train_dataset
        elif mode=='test':
            return test_dataset
        elif mode=='val':
            return val_dataset

# train=GenerateDate('train')
# loader=DataLoader(train,batch_size=32,shuffle=True)
#
# model=BertClassifier()
# for inputs,label in loader:
#     inputs_ids=inputs['input_ids'].squeeze(1)
#     masks=inputs['attention_mask']
#     print(inputs_ids.size())
#     print(masks.size())
#     output=model(inputs_ids,masks)
#     print(output.size())
#     print(output)
#     break
