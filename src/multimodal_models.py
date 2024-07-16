import torch
import torch.nn as nn
import torch.nn.functional as F

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

def two_step_loss(logits):
    with torch.no_grad():
        qx = F.softmax(logits, dim=1)
        qy = F.softmax(logits, dim=0)
        sx = qx.sum(dim=0)
        sy = qy.sum(dim=1)

    identity = torch.diagonal(logits).sum()
    cx = (logits * qx).sum()
    cy = (logits * qy).sum()
    cross =  0.5 * (logits * qx).sum(dim=1) @ sy + 0.5 * (logits * qy).sum(dim=0) @ sx

    return -(identity - cx - cy + cross)

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
    elif loss == "two_step":
        return two_step_loss
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