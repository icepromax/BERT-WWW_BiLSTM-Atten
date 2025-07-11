import pandas as pd
import re

def clean_text(text):
    if pd.isna(text):
        return ""
    # 去除换行符、制表符、多余空格
    text = re.sub(r'\s+', ' ', text)
    # 去除重复标点，如多个句号、感叹号、问号
    text = re.sub(r'([。！？,.!?])\1+', r'\1', text)
    # 去除无用标点，只保留中文常用符号
    text = re.sub(r'[【】<>「」『』（）()\[\]{}《》~@#￥%^&*_=|“”‘’\'\"/\\]', '', text)
    # 去除开头和结尾空格
    return text.strip()

def preprocess_reviews(input_csv, output_csv, text_column='review'):
    # 读取CSV
    df = pd.read_csv(input_csv)

    # 确保列存在
    if text_column not in df.columns:
        raise ValueError(f"列名 '{text_column}' 不存在于 CSV 文件中。")

    # 清洗评论列
    df[text_column] = df[text_column].apply(clean_text)

    # 保存到新文件
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"✅ 清洗完成，保存为：{output_csv}")

# 用法示例
if __name__ == '__main__':
    preprocess_reviews(
        input_csv='./data/ChnSentiCorp_htl_all.csv',
        output_csv='./data/cleaned_ChnSentiCorp_htl_all.csv',
        text_column='text'  # 如果你的列名是“评论内容”或其他，修改这里
    )
