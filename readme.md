# 🌟 BERT-WWM + BiLSTM + Attention 中文情感分析

本项目基于 **BERT-WWM + BiLSTM + 参数化注意力机制**，完成了对中文文本情感的分类任务。我们使用了两个经典中文情感数据集（**ChnSentiCorp** 和 **waimai_10k**），并通过消融实验评估了不同模型组件的效果。

------

## 🔍 模型架构说明

- **BERT**：中文 Whole Word Masking 预训练语言模型（RoBERTa-wwm-ext）
- **Bi-LSTM**：增强对上下文序列信息的建模
- **Parametric Attention**：引入可学习参数进行加权注意力机制

------

## 📊 消融实验结果对比

| 模型                   | 数据集       | 精确率（Precision） | 召回率（Recall） | F1 值（F1-Score） | 准确率（Accuracy） |
| ---------------------- | ------------ | ------------------- | ---------------- | ----------------- | ------------------ |
| BERT                   | waimai_10k   | 0.9157              | 0.9145           | 0.9149            | 0.9145             |
| BERT + Bi-LSTM         | waimai_10k   | 0.9185              | 0.9174           | 0.9178            | 0.9174             |
| BERT + Bi-LSTM + Atten | waimai_10k   | **0.9211**          | **0.9212**       | **0.9211**        | **0.9212**         |
| BERT                   | ChnSentiCorp | 0.8961              | 0.8970           | 0.8958            | 0.8970             |
| BERT + Bi-LSTM         | ChnSentiCorp | 0.8994              | 0.9003           | 0.8996            | 0.9003             |
| BERT + Bi-LSTM + Atten | ChnSentiCorp | **0.9058**          | **0.9062**       | **0.9058**        | **0.9062**         |



------

## 🧪 在线界面测试（FastAPI）

我们还使用 **FastAPI** 构建了一个情感分析 REST 接口，并配套开发了本地测试界面。支持实时输入中文文本，返回预测情感类别及置信度。

### 🖼️ 示例界面截图

![image-20250711094849801](C:\Users\86182\AppData\Roaming\Typora\typora-user-images\image-20250711094849801.png)

## 其它相关说明

需从huggingface下载chinese-roberta-wwm-ext预训练模型到RoBERTa_wwm_ext目录下
https://huggingface.co/hfl/chinese-roberta-wwm-ext



### 相关依赖

pip install -r requirements.txt 可直接下载

python==3.9

基础Python依赖

numpy==1.24.4
pandas==1.5.3
matplotlib==3.7.1
seaborn==0.12.2

机器学习/深度学习相关

torch==1.13.1
transformers==4.33.2
scikit-learn==1.2.2

FastAPI + ASGI服务器

fastapi==0.100.0
uvicorn==0.23.2

其他辅助工具

tqdm==4.65.0



对于前端页面，在前端文件夹下利用npm install 下载相关依赖

利用 npm run dev启动