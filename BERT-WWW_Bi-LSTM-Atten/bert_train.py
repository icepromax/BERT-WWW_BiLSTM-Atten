import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from bert_get_data_model import GenerateDate,BertClassifier1,BertClassifier,BertClassifierWithAttention
import matplotlib.pyplot as plt
from datetime import datetime


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True#强制 cuDNN 只使用确定性算法，保证相同输入和模型参数时，输出结果完全一致。

def save_model(save_name):
    if not os.path.exists(save_path):
        os.makedirs((save_path))
    torch.save(model.state_dict(),os.path.join(save_path,save_name))

save_path='./bert_checkpoint_crop'
save_image_path="./crop_image"
path='_atten'
if path=='_atten':
    model = BertClassifierWithAttention()
elif path=='_1':
    model = BertClassifier1()
else:
    model=BertClassifier()


epoch=8
batch_size=32

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random_seed=42
setup_seed(random_seed)

#定义模型

lr=1e-5
criterion=nn.CrossEntropyLoss()
optimizer=Adam(model.parameters(),lr=lr)


model=model.to(device)
criterion=criterion.to(device)

#构建数据集
train_datatset=GenerateDate("train")
dev_dataset=GenerateDate("val")
train_loader=DataLoader(train_datatset,batch_size=batch_size,shuffle=True)
dev_loader=DataLoader(dev_dataset,batch_size=batch_size)

best_dev_acc=0
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=len(train_loader)*1,  # 1个epoch为一个周期
    eta_min=1e-7  # 最小学习率
)
#定义画图所需参数
losses_train = []
Acc_train = []
losses_val = []
Acc_val = []

for epoch_num in range(epoch):
    model.train()
    total_acc_train=0
    total_loss_train=0

    for inputs,labels in tqdm(train_loader):
        input_ids=inputs['input_ids'].squeeze(1).to(device) #[32,35]
        masks=inputs['attention_mask'].to(device)#[32,1,35]
        labels=labels.to(device)
        output=model(input_ids,masks)
        bacth_loss=criterion(output,labels)
        bacth_loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        acc =(output.argmax(dim=1)==labels).sum().item()

        total_acc_train+=acc
        total_loss_train+=bacth_loss.item()



    #模型验证
    model.eval()
    total_acc_val=0
    total_loss_val=0


    with torch.no_grad():
        for inputs,labels in dev_loader:
            input_ids=inputs['input_ids'].squeeze(1).to(device)
            masks=inputs['attention_mask'].to(device)
            labels=labels.to(device)
            output=model(input_ids,masks)
            batch_loss=criterion(output,labels)
            acc=(output.argmax(dim=1)==labels).sum().item()
            total_acc_val+=acc
            total_loss_val+=batch_loss.item()

        loss_train=total_loss_train/len(train_datatset)
        acc_train=total_acc_train/len(train_datatset)
        loss_val=total_loss_val/len(dev_dataset)
        acc_val=total_acc_val / len(dev_dataset)

        #记录数据，用于绘制图像
        losses_train.append(loss_train)
        Acc_train.append(acc_train)
        losses_val.append(loss_val)
        Acc_val.append(acc_val)


        print(f'''Epochs:{epoch_num+1}
            | Data Time:{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            | Train Loss:{loss_train:.3f}
            | Train Accuracy:{acc_train:.3f}
            | Val Loss:{loss_val:.3f}
            | Val Accuracy:{acc_val:.3f}
            ''')
        #保存最优模型

        if acc_val>best_dev_acc:
            best_dev_acc=acc_val
            save_model(f'best{path}.pt')

epoches=list(range(1,len(losses_train)+1))

os.makedirs(save_image_path, exist_ok=True)  # 确保目录存在

plt.plot(epoches,losses_train,label="Train losses")
plt.plot(epoches,losses_val,label="Val losses")
plt.xlabel("epoch")
plt.ylabel("losses")
plt.title("The Losses Plot")
plt.legend()
plt.savefig(os.path.join(save_image_path,f"base_losses{path}"),bbox_inches="tight",dpi=300)
plt.show()
plt.close()

save_image_path="./waimai_image"
plt.plot(epoches,Acc_train,label="Train Acc")
plt.plot(epoches,Acc_val,label="Val Acc")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("The Accuracy Plot")
plt.legend()
plt.savefig(os.path.join(save_image_path,f"base_acc{path}"),bbox_inches="tight",dpi=300)
plt.show()
plt.close()
save_model(f'last{path}.pt')












