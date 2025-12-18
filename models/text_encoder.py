import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class TextEncoder(nn.Module):
    def __init__(self, hidden_dim=512, freeze_layers=6):
        super(TextEncoder, self).__init__()
        
        model_name = 'DeepChem/ChemBERTa-zinc-base-v1'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        
        self.bert_hidden_dim = self.bert.config.hidden_size
        self.projection = nn.Linear(self.bert_hidden_dim, hidden_dim)
        
    def forward(self, text_prompts):
        tokens = self.tokenizer(
            text_prompts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(self.bert.device)
        
        outputs = self.bert(**tokens)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        text_embedding = self.projection(cls_embedding)
        
        return text_embedding
