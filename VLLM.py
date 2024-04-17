import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

class VLLM:
    def __init__(self, model_name='openbmb/MiniCPM-V-2'):
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device=self.device, dtype=torch.bfloat16)

    def infer(self, img_path, question, sampling=True, temperature=0.7):
        image = Image.open(img_path).convert('RGB')
        msgs = [{'role': 'user', 'content': question}]

        res, _, _ = self.model.chat(
            image=image,
            msgs=msgs,
            context=None,
            tokenizer=self.tokenizer,
            sampling=sampling,
            temperature=temperature
        )
        return res

# Usage example:
# vllm = VLLM()

# while True:
#     img_path = "nami.png"
#     question = input('Enter question: ')
#     result = vllm.infer(img_path, question)
#     print(result)

# ----
# result = llm_inference.infer('nami.png', 'What is in the image?')
# print(result)

# result = llm_inference.infer('nami.png', 'Is this a boy or girl in the image?')
# print(result)


