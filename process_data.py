import csv
import json
from pathlib import Path
import pandas as pd
def csv_to_json_converter(csv_file_path, json_file_path):

    # 确保输出目录存在
    Path(json_file_path).parent.mkdir(parents=True, exist_ok=True)

    data = []
    try:
        df = pd.read_csv(csv_file_path,sep='|')

        for row in df.values:
            # 格式化 question 和 answer 内容
            formatted_question = "<|im_start|>user\ndescribe this picture.<|im_end|>"
            formatted_answer = f"<|im_start|>assistant\n{row[2]}<|im_end|>"
            
            # 构建 conversations 列表
            conversations = [
                {"from": "user", "value": formatted_question},
                {"from": "assistant", "value": formatted_answer}
            ]
            
            # 创建最终的 JSON 对象
            json_object = {
                "conversations": conversations,
                "image": row[0]
            }
            
            data.append(json_object)
    
        # 将收集到的数据写入 JSON 文件
        with open(json_file_path, mode='w', encoding='utf-8') as json_file:
            # 使用 ensure_ascii=False 确保中文等非 ASCII 字符正常显示
            # indent=2 让输出的 JSON 更易读
            json.dump(data, json_file, ensure_ascii=False, indent=2)
            
        print(f"✅ 转换成功！文件已保存到: {json_file_path}")

    except FileNotFoundError:
        print(f"❌ 错误: 找不到 CSV 文件 '{csv_file_path}'")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")

if __name__ == '__main__':
    # 假设你的 CSV 文件名为 'data.csv'，并且和这个 Python 脚本在同一目录下
    input_csv_path = './results.csv'
    # 输出的 JSON 文件名
    output_json_path = './frickr30k.json'
    # 调用函数进行转换
    csv_to_json_converter(input_csv_path, output_json_path)