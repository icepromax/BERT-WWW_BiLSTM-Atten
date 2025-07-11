# BERT-WWW_BiLSTM-Atten
利用BERT-WWW-BiLSTM-Atten进行了中文情感分析的研究，利用消融实验进行对比，并且构造一个简易界面测试了模型
消融实验结果汇总表
模型	数据集	精确率（Precision）	召回率（Recall）	F1 值（F1-Score）	准确率（Accuracy）
BERT	waimai_10k	0.9157	0.9145	0.9149	0.9145
BERT+Bi-LSTM	waimai_10k	0.9185	0.9174	0.9178	0.9174
BERT+Bi-LSTM+Attention	waimai_10k	0.9211	0.9212	0.9211	0.9212
BERT	ChnSentiCorp	0.8961	0.8970	0.8958	0.8970
BERT +Bi-LSTM	ChnSentiCorp	0.8994	0.9003	0.8996	0.9003
BERT +Bi-LSTM +Attention	ChnSentiCorp	0.9058	0.9062	0.9058	0.9062

界面测试
<img width="866" height="434" alt="image" src="https://github.com/user-attachments/assets/4247acd3-e7bc-4eb1-addc-0a48b45868cf" />


