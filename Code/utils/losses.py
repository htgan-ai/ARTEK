import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropy(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
    def forward(self, logits, targets):
        return self.ce(logits, targets)

class NLLLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.nll = nn.NLLLoss(weight=weight)
    def forward(self, logits, targets):
        loss = self.nll(F.log_softmax(logits, -1), targets)
        return loss

class CrossEntropyWithLS(nn.Module):
    def __init__(self, label_smoothing: float = 0.1, weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=weight)
    def forward(self, logits, targets):
        return self.ce(logits, targets)

class FocalCrossEntropy(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # None | float | Tensor[num_classes]
        self.label_smoothing = label_smoothing
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)  # = prob of true class
        loss = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            if isinstance(self.alpha, float):
                a = torch.full_like(targets, fill_value=self.alpha, dtype=torch.float)
            else:
                a = self.alpha.to(logits.device)[targets]
            loss = a * loss
        return loss.mean()

class BalancedSoftmaxCE(nn.Module):
    """Implements Eq.(3) of Balanced Softmax: φ̂_j = n_j e^{z_j} / sum_i n_i e^{z_i}."""
    def __init__(self, class_counts):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        counts = torch.clamp(counts, min=1.0)
        self.register_buffer('log_counts', counts.log())
    def forward(self, logits, targets):
        logits_bs = logits + self.log_counts.unsqueeze(0)  # add log(n_j) to each class logit
        return F.cross_entropy(logits_bs, targets)

class LabelAwareSmoothingCE(nn.Module):
    def __init__(self, class_counts, base_eps: float = 0.05, beta: float = 0.5, max_eps: float = 0.3):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        counts = torch.clamp(counts, min=1.0)
        n_max = counts.max()
        eps = base_eps * (n_max / counts).pow(beta)
        eps = torch.clamp(eps, 0.0, max_eps)
        self.register_buffer('eps', eps)
    def forward(self, logits, targets):
        logp = F.log_softmax(logits, dim=1)
        n, c = logits.size()
        with torch.no_grad():
            y = torch.zeros_like(logp)
            y.scatter_(1, targets.view(-1,1), 1.0)
            e = self.eps[targets].unsqueeze(1)  # per-sample epsilon
            y_smooth = (1 - e) * y + e / c
        loss = -(y_smooth * logp).sum(dim=1).mean()
        return loss

class Poly1CrossEntropy(nn.Module):
    """PolyLoss Poly-1: CE + epsilon * (1 - p_t)"""
    def __init__(self, epsilon: float = 1.0):
        super().__init__()
        self.epsilon = epsilon
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = F.softmax(logits, dim=1)[torch.arange(logits.size(0), device=logits.device), targets]
        return (ce + self.epsilon * (1 - pt)).mean()
    

class SupConFromEmbedding(nn.Module):
    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = float(temperature)
        self.base_temperature = float(base_temperature)

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = F.normalize(emb, dim=1)
        sim = emb @ emb.t() / self.temperature            # [B,B]
        B = emb.size(0)
        device = emb.device

        mask_pos = labels.view(-1, 1).eq(labels.view(1, -1)).float().to(device)
        logits = sim - torch.eye(B, device=device) * 1e9
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        pos_cnt = mask_pos.sum(1).clamp_min(1.0)
        mean_log_prob_pos = (mask_pos * log_prob).sum(1) / pos_cnt

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()
    
class CircleLoss(nn.Module):
    """
    Circle Loss（CVPR'20）
    """
    def __init__(self, m: float = 0.25, gamma: float = 256.0):
        super().__init__()
        self.m = float(m)
        self.gamma = float(gamma)

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = F.normalize(emb, dim=1)
        sim = emb @ emb.t()                                
        labels = labels.view(-1, 1)
        mask_pos = labels.eq(labels.t()).float()
        mask_neg = 1.0 - mask_pos

        sp = sim * mask_pos
        sn = sim * mask_neg
        ap = torch.clamp_min(-sp + 1.0 + self.m, 0.0)
        an = torch.clamp_min(sn + self.m, 0.0)

        logit_p = -self.gamma * ap * (sp - (1.0 - self.m))
        logit_n =  self.gamma * an * (sn - self.m)

        neg_term = torch.where(
            (mask_neg.sum(1) > 0),
            torch.logsumexp(logit_n + (1 - mask_neg) * (-1e9), dim=1),
            torch.zeros_like(sim[:, 0])
        )
        pos_term = torch.where(
            (mask_pos.sum(1) > 0),
            torch.logsumexp(logit_p + (1 - mask_pos) * (-1e9), dim=1),
            torch.zeros_like(sim[:, 0])
        )
        loss = F.softplus(neg_term + pos_term).mean()
        return loss
    
class ProxyAnchorLoss(nn.Module):
    """
    Proxy-Anchor Loss（CVPR'20）
    """
    def __init__(self, num_classes: int, feat_dim: int, alpha: float = 32.0, margin: float = 0.1):
        super().__init__()
        self.proxies = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.kaiming_normal_(self.proxies, nonlinearity='linear')
        self.alpha = float(alpha)
        self.margin = float(margin)

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = F.normalize(emb, dim=1)
        P   = F.normalize(self.proxies, dim=1)
        sim = emb @ P.t()                                    # [B,C]
        y   = torch.zeros_like(sim).scatter_(1, labels.view(-1, 1), 1.0)

        pos = sim[y.bool()]
        neg = sim[~y.bool()]

        pos_term = torch.log1p(torch.exp(-self.alpha * (pos - (1.0 - self.margin)))).mean() \
                   if pos.numel() > 0 else sim.new_tensor(0.0)
        neg_term = torch.log1p(torch.exp( self.alpha * (neg - (-self.margin)))).mean() \
                   if neg.numel() > 0 else sim.new_tensor(0.0)
        return pos_term + neg_term

class MagFaceRegularizer(nn.Module):
    def __init__(self, lower: float = 10.0, upper: float = 110.0, lam: float = 1e-3):
        super().__init__()
        self.lower = float(lower)
        self.upper = float(upper)
        self.lam = float(lam)

    def forward(self, raw_emb: torch.Tensor) -> torch.Tensor:

        r = raw_emb.norm(p=2, dim=1)
        reg = ((r - self.lower).clamp_min(0) ** 2 + (r - self.upper).clamp_max(0) ** 2).mean()
        return self.lam * reg

def build_criterion(name: str, num_classes: int, class_counts=None, class_weight=None):
    name = name.lower()
    if name == 'ce':
        return CrossEntropy()
    if name == 'nll':
        return NLLLoss(weight=class_weight)
    if name == 'ce_ls':
        return CrossEntropyWithLS(0.1)
    if name == 'focal':
        return FocalCrossEntropy(gamma=2.0, label_smoothing=0.05)
    if name == 'balanced_softmax':
        assert class_counts is not None, "class_counts needed for BalancedSoftmax"
        return BalancedSoftmaxCE(class_counts)
    if name == 'las':
        assert class_counts is not None, "class_counts needed for Label-Aware Smoothing"
        return LabelAwareSmoothingCE(class_counts, base_eps=0.05, beta=0.5)
    if name == 'poly1':
        return Poly1CrossEntropy(epsilon=1.0)
    if name == 'supcon':
        return SupConFromEmbedding(temperature=0.07, base_temperature=0.07)
    if name == 'circle':
        return CircleLoss(m=0.25, gamma=256.0)
    if name == 'proxyanchor':
        return ProxyAnchorLoss(num_classes=num_classes, feat_dim=512, alpha=32.0, margin=0.1)
    if name == 'magface_reg':
        return MagFaceRegularizer(lower=10.0, upper=110.0, lam=1e-3)
    raise ValueError(f'Unknown loss: {name}')
