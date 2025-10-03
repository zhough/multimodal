from model import VLMModel
from transformers import AutoConfig,AutoModelForCausalLM
from vision_config import VisionConfig
import torch
from torch.utils.data import Dataset,DataLoader
vconfig = VisionConfig()
from transformers import AutoImageProcessor
from PIL import Image
import torchvision.transforms as transforms

class Config():
    def __init__(self):
        self.epochs = 10
        self.epochs = 5
        self.batch_size = 8
        self.learning_rate = 6e-5
        self.weight_decay = 1e-4
        self.model_save_path = './output/model.pth'

config = Config()
image_processor = AutoImageProcessor.from_pretrained(vconfig.model_name)

class VLMDataSet(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data  # 格式: [{"image_path": "...", "text": "..."}, ...]
        self.tokenizer = tokenizer
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 根据ViT模型要求调整
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 处理图像
        image = Image.open(item["image_path"]).convert("RGB")
        pixel_values = image_processor(image, return_tensors="pt")["pixel_values"].squeeze(0)  # [C, H, W]
        
        # 处理文本（以文本生成为例，label与input_ids一致，padding部分mask）
        text = item["text"]
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # 忽略padding部分的loss
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def init_model(tokenizer,trained_model=None,rank=0):
    lconfig = AutoConfig.from_pretrained(vconfig.llm)
    model = VLMModel(lconfig)
    model.to(rank)
    if trained_model is None:
        llm = AutoModelForCausalLM.from_pretrained(vconfig.llm)
        model.load_state_dict(
            llm.state_dict(), # 官方模型的权重在 .model 属性里
            strict=False
        )
    else:
        # 加载权重文件（指定 map_location 到当前 GPU）
        device = torch.device('cuda', rank)
        state_dict = torch.load(trained_model, map_location=device)
        
        # 移除所有权重键的 module. 前缀
        new_state_dict = {}
        for key, value in state_dict.items():
            # 去掉键开头的 module.（若存在）
            if key.startswith('module.'):
                new_key = key[len('module.'):]  # 从第 7 个字符开始截取（module. 共 7 个字符）
            else:
                new_key = key  # 若没有前缀，直接保留原键
            new_state_dict[new_key] = value
        
        # 2.3 加载处理后的权重到模型
        model.load_state_dict(new_state_dict)