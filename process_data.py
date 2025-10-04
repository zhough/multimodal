import csv
import json
from pathlib import Path

def csv_to_json_converter(csv_file_path, json_file_path):
    """
    将指定的 CSV 文件转换为特定格式的 JSON 文件。

    CSV 格式 (三列):
    question,answer,image_id

    转换后的 JSON 格式 (一个对象列表):
    [
      {
        "conversations": [
          {"from": "user", "value": "<|im_start|>user\n[question_content]<|im_end|>"},
          {"from": "assistant", "value": "<|im_start|>assistant\n[answer_content]<|im_end|>"}
        ],
        "image": "[image_id]"
      },
      ...
    ]
    """
    # 确保输出目录存在
    Path(json_file_path).parent.mkdir(parents=True, exist_ok=True)

    data = []
    try:
        with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            
            # 检查 CSV 是否包含所需的列
            required_columns = {'question', 'answer', 'image_id'}
            if not required_columns.issubset(csv_reader.fieldnames):
                missing = required_columns - set(csv_reader.fieldnames)
                raise ValueError(f"CSV 文件缺少必要的列: {', '.join(missing)}")

            for row in csv_reader:
                # 格式化 question 和 answer 内容
                formatted_question = f"<|im_start|>user\n{row['question']}<|im_end|>"
                formatted_answer = f"<|im_start|>assistant\n{row['answer']}<|im_end|>"
                
                # 构建 conversations 列表
                conversations = [
                    {"from": "user", "value": formatted_question},
                    {"from": "assistant", "value": formatted_answer}
                ]
                
                # 创建最终的 JSON 对象
                json_object = {
                    "conversations": conversations,
                    "image": row['image_id']
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
    input_csv_path = './dataset/data_eval.csv'
    # 输出的 JSON 文件名
    output_json_path = './data_val.json'
    # 调用函数进行转换
    csv_to_json_converter(input_csv_path, output_json_path)