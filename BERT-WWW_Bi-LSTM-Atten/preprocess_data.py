import re
import pandas as pd


def clean_social_media_text(text):
    """
    清洗社交媒体文本的专用函数
    """
    if not isinstance(text, str):
        return ""

    # 去除微博表情符号[xxx]
    text = re.sub(r'\[.*?\]', '', text)

    # 去除@提及
    text = re.sub(r'@[\w\u4e00-\u9fff_-]+', '', text)

    # 去除URL链接
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    # 去除转发标记(//, ///, //@等变体)
    text = re.sub(r'/+@?\s*', '', text)

    # 去除话题标签但保留文字内容
    text = re.sub(r'#([^#]+)#', r'\1', text)

    # 中文冒号（：）
    text = re.sub(r'[^\s]：', '', text)  # 匹配非空白字符+中文冒号
    # 英文冒号（:）
    text = re.sub(r'[^\s]:', '', text)  # 匹配非空白字符+英文冒号

    # 删除所有中文冒号（：）和英文冒号（:）
    text = re.sub(r'[：:]', '', text)

    # 处理中文和英文标点
    all_punctuation = r'[，。、；：？！「」『』"".''“”‘’—…()（）【】{}\[\]〈〉《》]'

    # 1. 删除最前面的标点符号
    text = re.sub(f'^{all_punctuation}+', '', text.strip())

    # 2. 删除与其他内容不相连的标点符号（前后有空格）
    text = re.sub(f' {all_punctuation}+ ', ' ', text)  # 前后都有空格
    text = re.sub(f' {all_punctuation}+$', '', text)  # 前面有空格，后面是结尾
    text = re.sub(f'^{all_punctuation}+ ', '', text)  # 前面是开头，后面有空格

    # 3. 删除连续的标点符号（保留一个）
    text = re.sub(f'({all_punctuation}){2,}', r'\1', text)

    # 统一省略号
    text = re.sub(r'…{2,}', '...', text)

    # 去除多余空格和换行
    text = ' '.join(text.split())

    return text.strip()


def preprocess_social_data(input_file, output_file):
    """
    预处理社交媒体数据
    """
    # 读取数据
    df = pd.read_csv(input_file)

    # 清洗文本
    df['cleaned_review'] = df['review'].apply(clean_social_media_text)

    # 过滤空评论和过短评论
    df = df[df['cleaned_review'].str.len() > 3]

    # 保存结果
    df[['label', 'cleaned_review']].to_csv(
        output_file,
        index=False,
        encoding='utf-8-sig',
        header=['label', 'text']
    )
    print(f"清洗完成，结果保存到 {output_file}")


# 使用示例
input_file = 'data/weibo_senti_100k.csv'
output_file = 'data/cleaned_weibo_senti_100k.csv'
preprocess_social_data(input_file, output_file)