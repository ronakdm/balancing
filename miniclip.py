import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, BertForMaskedLM

MODEL_DIR = "~/balancing/models" # TODO: Change this to where the models are on your machine!

####################################
# MODEL DEFINITIONS
####################################

class MLP(nn.Module):
    def __init__(self, in_features, hidden_size, n_layers, out_features):
        super(MLP, self).__init__()
        if n_layers > 0:
            self.proj = nn.Linear(in_features, hidden_size)
            self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(n_layers - 1)])
            self.out = nn.Linear(hidden_size, out_features)
        else:
            self.out = nn.Linear(in_features, out_features)
        self.n_layers = n_layers

    def forward(self, x):
        if self.n_layers > 0:
            x = F.relu(self.proj(x))
            for layer in self.layers:
                x = F.relu(layer(x))
        return self.out(x)


def jointly_centered_loss(logits):
    norm_factor = torch.logsumexp(torch.flatten(logits), dim=0)
    return -torch.mean(torch.diagonal(logits) - norm_factor)

def clip_loss(logits):
    cx   = F.log_softmax(logits, dim=1)
    cy   = F.log_softmax(logits, dim=0)
    return -torch.mean(0.5 * torch.diagonal(cx) + 0.5 * torch.diagonal(cy))

def doubly_centered_loss(logits):
    cx   = F.log_softmax(logits, dim=1)
    cy   = F.log_softmax(logits, dim=0)
    cycx = F.log_softmax(cx, dim=0)
    cxcy = F.log_softmax(cy, dim=1)
    return -torch.mean(0.5 * torch.diagonal(cycx) + 0.5 * torch.diagonal(cxcy))
    
def get_loss(loss):
    if loss == "clip":
        return clip_loss
    elif loss == "jointly_centered":
        return jointly_centered_loss
    elif loss == "doubly_centered":
        return doubly_centered_loss
    else:
        raise NotImplementedError
    
class MiniCLIP(nn.Module):
    def __init__(
            self,
            in_features_img, 
            hidden_size_img, 
            n_layers_img,
            in_features_txt, 
            hidden_size_txt, 
            n_layers_txt,
            out_features, 
            loss="clip",
            architecture="miniclip"
        ):
        if architecture != "miniclip":
            raise ValueError(
                f"Incorrect architecture specification '{architecture}' for model MiniCLIP!"
            )
        super(MiniCLIP, self).__init__()
        self.image_encoder = MLP(in_features_img, hidden_size_img, n_layers_img, out_features)
        self.text_encoder  = MLP(in_features_txt, hidden_size_txt, n_layers_txt, out_features)
        # self.logit_scale = torch.nn.Parameter(0.1 * torch.randn(1)) # learnable parameter
        self.scale = 100.
        self.loss = get_loss(loss)

    def forward(self, x, y, sample_weight=None):

        # extract feature representations of each modality
        I_f = self.image_encoder(x)
        T_f = self.text_encoder(y)

        # joint multimodal embedding [n, d_e]
        I_e = F.normalize(I_f)
        T_e = F.normalize(T_f)

        # scaled pairwise cosine similarities [n, n]
        logits = torch.matmul(I_e, T_e.T) * self.scale

        # symmetric loss function
        loss = self.loss(logits)

        return loss, logits
    
####################################
# WRAPPERS FOR EVALUATION
####################################

class ModelWrapper(nn.Module):

    def __init__(self, foundation_img, foundation_txt, head):
        super(ModelWrapper, self).__init__()
        self.foundation_img = foundation_img
        self.foundation_txt = foundation_txt
        self.head = head
        self.scale = 100.

    def forward(self, image, text):
        I_f = self.encode_image(image)
        T_f = self.encode_text(text)
        I_e = F.normalize(I_f)
        T_e = F.normalize(T_f)
        logits = torch.matmul(I_e, T_e.T) * self.scale

        return logits, logits.T

    
    def encode_image(self, image):
        image_features = self.foundation_img.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return self.head.image_encoder(image_features)

    def encode_text(self, text):
        text_features = self.foundation_txt.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return self.head.text_encoder(text_features)
    
def load_head_model(model_name):
    if "gpt" in model_name or "bert" in model_name:
        model_cfg = {
            "architecture": "miniclip",
            "in_features_img": 512,
            "hidden_size_img": 256,
            "n_layers_img": 2,
            "in_features_txt": 768,
            "hidden_size_txt": 256,
            "n_layers_txt": 2,
            "out_features": 128,
            "loss": "two_step",
        }
        model = MiniCLIP(**model_cfg)
    else:
        model_cfg = {
            "architecture": "miniclip",
            "in_features_img": 512,
            "hidden_size_img": 256,
            "n_layers_img": 2,
            "in_features_txt": 512,
            "hidden_size_txt": 256,
            "n_layers_txt": 2,
            "out_features": 128,
            "loss": "clip", # does not matter for evaluation
        }
        model = MiniCLIP(**model_cfg)

    output_dir = MODEL_DIR
    model.load_state_dict(torch.load(os.path.join(output_dir, f"{model_name}.pt")))
    return model

class SequenceModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokenized_texts):
        # dummy method, as only encode_text should be used
        return self.encode_text(tokenized_texts)

    def encode_text(self, tokenized_texts):
        device = tokenized_texts.get_device()
        input_ids, attn_mask = tokenized_texts[0], tokenized_texts[1]
        lens = torch.tensor([mask.sum().item() for mask in attn_mask])
        out = self.model(input_ids=input_ids.to(device), attention_mask=attn_mask.to(device))
        return torch.stack([out["hidden_states"][-1][i, 0:lens[i], :].mean(dim=0) for i in range(len(input_ids))])

class TokenizerWrapper:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, texts):
        # texts is a list of strings
        encoded_input = self.tokenizer(
            texts, 
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=32,
            # pad_to_max_length=True,
            truncation=True,
            padding='max_length',
            return_tensors="pt",  # Return pytorch tensors.return_tensors='pt'
            return_attention_mask=True,
        )
        # stack into two tensors for which the first are the ids and the second is the attention mask
        return torch.stack([encoded_input["input_ids"], encoded_input["attention_mask"]])

####################################
# LOADING FUNCTION
####################################

def load_miniclip(
        model_name: str = 'miniclip', 
        pretrained: str = 'laion2b_s34b_b79k', 
        cache_dir: str = None, 
        device="cpu",
    ):
    head_model = load_head_model(model_name)
    if "gpt" in model_name:
        seq_model = GPT2LMHeadModel.from_pretrained(f'gpt2', output_hidden_states=True)
        tokenizer = GPT2Tokenizer.from_pretrained(f'gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        tokenizer = TokenizerWrapper(tokenizer)
        foundation_img, _, transform = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', cache_dir=cache_dir)
        foundation_txt = SequenceModelWrapper(seq_model)
        model = ModelWrapper(foundation_img, foundation_txt, head_model).to(device)
    elif "bert" in model_name:
        seq_model  = BertForMaskedLM.from_pretrained(
            "bert-base-uncased",
            output_attentions=False,
            output_hidden_states=True,
        )
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        tokenizer = TokenizerWrapper(tokenizer)
        foundation_img, _, transform = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', cache_dir=cache_dir)
        foundation_txt = SequenceModelWrapper(seq_model)
        model = ModelWrapper(foundation_img, foundation_txt, head_model).to(device)
    else:
        foundation_img, _, transform = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', cache_dir=cache_dir)
        foundation_txt, _, transform = open_clip.create_model_and_transforms('ViT-B-32', pretrained='datacomp_xl_s13b_b90k', cache_dir=cache_dir)
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        model = ModelWrapper(foundation_img, foundation_txt, head_model).to(device)
    return model, transform, tokenizer