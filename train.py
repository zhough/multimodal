from model import VLMModel
from transformers import AutoConfig
from vision_config import VisionConfig
vconfig = VisionConfig()
config = AutoConfig.from_pretrained(vconfig.llm)
model = VLMModel(config)
for name, param in model.named_parameters():
    # 打印参数名和其形状
    print(f"名称: {name:<80} | 形状: {param.shape}")