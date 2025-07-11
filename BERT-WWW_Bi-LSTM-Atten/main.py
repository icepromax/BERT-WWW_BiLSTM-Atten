import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import torch
from transformers import BertTokenizer
from bert_get_data_model import BertClassifierWithAttention
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 定义请求模型
class AnalysisRequest(BaseModel):
    text: str
    dataset: str  # 数据集类型: chn 或 waimai


# 初始化应用
app = FastAPI(title="中文情感分析API", version="1.0")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模型配置
MODEL_CONFIG = {
    "chn": {
        "model_path": "./bert_checkpoint_crop/best_atten.pt",
        "bert_name": "./RoBERTa_wwm_ext",  # 修改为本地路径
        "labels": ["消极", "积极"],
        "max_length": 80  # 与本地调用保持一致
    },
    "waimai": {
        "model_path": "./bert_checkpoint_waimai/best_atten.pt",
        "bert_name": "./RoBERTa_wwm_ext",  # 修改为本地路径
        "labels": ["消极", "积极"],
        "max_length": 80  # 与本地调用保持一致
    }
}

# 全局模型缓存
model_cache = {}


def load_model(dataset_type: str):
    """加载模型并缓存"""
    if dataset_type in model_cache:
        return model_cache[dataset_type]

    config = MODEL_CONFIG.get(dataset_type)
    if not config:
        raise ValueError(f"未知的数据集类型: {dataset_type}")

    if not os.path.exists(config["model_path"]):
        raise FileNotFoundError(f"模型文件不存在: {config['model_path']}")

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")

    try:
        # 加载分词器
        tokenizer = BertTokenizer.from_pretrained(config["bert_name"])

        # 加载模型
        model = BertClassifierWithAttention()
        model.load_state_dict(torch.load(config["model_path"], map_location=device))
        model = model.to(device)
        model.eval()

        # 缓存模型
        model_cache[dataset_type] = (model, tokenizer, device, config["labels"], config["max_length"])
        return model_cache[dataset_type]

    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise


@app.on_event("startup")
async def startup_event():
    """预加载模型"""
    for dataset in MODEL_CONFIG.keys():
        try:
            load_model(dataset)
            logger.info(f"成功预加载 {dataset} 模型")
        except Exception as e:
            logger.error(f"预加载 {dataset} 模型失败: {str(e)}")


@app.get("/")
async def health_check():
    return {"status": "running", "models_loaded": list(model_cache.keys())}


@app.post("/api/analyze")
async def analyze_text(request: AnalysisRequest):
    """情感分析接口"""
    # 验证输入
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="输入文本不能为空")

    if request.dataset not in MODEL_CONFIG:
        raise HTTPException(status_code=400, detail="不支持的数据集类型")

    try:
        # 获取模型和配置
        model, tokenizer, device, labels, max_length = load_model(request.dataset)

        # 文本预处理 (与本地调用完全一致)
        bert_input = tokenizer(
            request.text,
            padding='max_length',
            max_length=max_length,  # 使用配置中的max_length
            truncation=True,
            return_tensors='pt'
        )

        # 打印输入维度 (调试用)
        logger.debug(f"Input ids size: {bert_input['input_ids'].size()}")
        logger.debug(f"Attention mask size: {bert_input['attention_mask'].size()}")

        # 移动到设备
        input_ids = bert_input['input_ids'].to(device)
        mask = bert_input['attention_mask'].unsqueeze(1).to(device)

        # 推理
        with torch.no_grad():
            output = model(input_ids, mask)
            probabilities = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1).item()
            confidence = probabilities[0][pred].item()

        return {
            "sentiment": labels[pred],
            "confidence": confidence,
            "model": MODEL_CONFIG[request.dataset]["bert_name"]
        }

    except torch.cuda.OutOfMemoryError:
        raise HTTPException(
            status_code=507,
            detail="GPU内存不足，请尝试缩短文本长度"
        )
    except Exception as e:
        logger.exception("分析过程中发生错误")
        raise HTTPException(
            status_code=500,
            detail=f"分析错误: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)