import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import LlamaConfig

def concrete_bernoulli(logits, temperature=0.1, hard=False):
    """
    Concrete relaxation of Bernoulli distribution.
    
    Args:
        logits (torch.Tensor): Logits of the Bernoulli distribution.
        temperature (float): Temperature parameter for the relaxation.
        hard (bool): If True, returns discrete samples. If False, returns relaxed samples.
        
    Returns:
        torch.Tensor: Relaxed or discrete samples from the Concrete Bernoulli distribution.
    """
    eps = 1e-7
    u = torch.rand_like(logits)
    y = torch.log(u + eps) - torch.log(1 - u + eps) + logits
    y = torch.sigmoid(y / temperature)
    
    if hard:
        y_hard = (y > 0.5).float()
        y = (y_hard - y).detach() + y
        
    return y

class PerturbLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.perturb_config = config.perturb_config
        self.no_alpha_net = getattr(self.perturb_config, "no_alpha_net", False)

        n_scale_net_layers= getattr(self.perturb_config, "n_scale_net_layers", 1)
        n_w_layers = getattr(self.perturb_config, "n_w_layers", 1)

        if n_w_layers == 1:
            self.w = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            layers = []
            for _ in range(n_w_layers - 1):
                layers += [nn.Linear(config.hidden_size, config.hidden_size), nn.ReLU()]
            layers += [nn.Linear(config.hidden_size, config.hidden_size)]
            self.w = nn.Sequential(*layers)

        if not self.no_alpha_net:
            self.alpha_net = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
        
        if n_scale_net_layers == 1:
            self.scale_net = nn.Linear(config.hidden_size, 1)
        else:
            layers = []
            for _ in range(n_scale_net_layers - 1):
                layers += [nn.Linear(config.hidden_size, config.hidden_size), nn.ReLU()]
            layers += [nn.Linear(config.hidden_size, 1)]
            self.scale_net = nn.Sequential(*layers)
        self.temperature = self.perturb_config.temperature
        self.dropout_prob = self.perturb_config.dropout_prob
        self.train_entity_mask = getattr(self.perturb_config, "train_entity_mask", False)
        self.no_gumbel = getattr(self.perturb_config, "no_gumbel", False)
        self.noise_type = getattr(self.perturb_config, "noise_type", "learnable")
        # print("Noise type:", self.noise_type)
        self.no_mask = getattr(self.perturb_config, "no_mask", False)
        self.no_sampling = getattr(self.perturb_config, "no_sampling", False)

    def generate_scale(self, h):
        scale_unnormalized = self.scale_net(h)
        if self.no_gumbel:
            scale = torch.sigmoid(scale_unnormalized) # Make it more "steep"
        else:
            scale = concrete_bernoulli(scale_unnormalized)
        return scale, scale_unnormalized

    def forward(
        self, 
        h, 
        apply_perturb, 
        attention_mask = None,
        entity_mask = None,
        aug_labels = None,
        noise_mask = None,
    ):
        mu = self.w(h) # Encoding with augmentation labels
              
        if entity_mask is not None:
            if not self.train_entity_mask:
                scale = (1.0 - entity_mask.to(dtype=h.dtype)).unsqueeze(-1) # Flip as 1 indicates the perturbed position
                scale_loss = torch.zeros(1, dtype=h.dtype, device=h.device)
            else:
                scale_pred, scale_unnormalized_pred = self.generate_scale(h)
                _scale_pred = torch.sigmoid(scale_unnormalized_pred.squeeze(-1)[attention_mask == 1])
                _scale_gt = (1.0 - entity_mask.to(dtype=scale_pred.dtype)[attention_mask == 1])

                loss_fct = nn.L1Loss()
                scale_loss = loss_fct(_scale_pred[_scale_gt == 0.0], _scale_gt[_scale_gt == 0.0])

                # Regulate the number of perturbations
                scale_percent_loss = torch.abs(
                    (torch.sigmoid(scale_unnormalized_pred.squeeze(-1)) * attention_mask).sum(dim=-1) \
                    - (attention_mask.sum(dim=-1) * self.dropout_prob)
                ).mean()
                scale_loss += scale_percent_loss
                # GT Scale for perturbation training
                # entity_mask = 1: entity for sentence (not perturb)
                # entity_mask = 0: non entity for sentence (perturb)
                scale = (1.0 - entity_mask.to(dtype=h.dtype)).unsqueeze(-1)
        else:
            scale, scale_unnormalized = self.generate_scale(h)
            scale_loss = torch.abs((scale.squeeze(-1) * attention_mask).sum(dim=-1) \
                - (attention_mask.sum(dim=-1) * self.dropout_prob)).mean()
        
        # Compute z using softplus activation function
        
        if self.noise_type == "uniform":
            dims = attention_mask.sum(-1) * h.size(2)
            mag_norm = 5.0 / torch.sqrt(dims)
            z = torch.zeros_like(h).uniform_(-1, 1) * mag_norm.unsqueeze(-1).unsqueeze(-1)
            if self.no_mask:
                perturb = z
            else:
                perturb = torch.ones_like(z) * (1.0 - scale) + z * scale
            new_h = h + perturb.to(dtype=h.dtype)
        elif self.noise_type == "gaussian":
            alpha = torch.randn_like(h)
            z = F.softplus(alpha / self.temperature, beta=0.693, threshold=2.0)
            if self.no_mask:
                perturb = z
            else:
                perturb = torch.ones_like(z) * (1.0 - scale) + z * scale
            new_h = h * perturb
        elif self.noise_type == "gaussian_additive":
            z = torch.randn_like(h)
            if self.no_mask:
                perturb = z
            else:
                perturb = torch.ones_like(z) * (1.0 - scale) + z * scale
            new_h = h + perturb
        elif self.noise_type == "learnable_scale":
            dims = attention_mask.sum(-1) * h.size(2)
            mag_norm = 5.0 / torch.sqrt(dims) 
            alpha = mu + (torch.randn_like(mu) * mag_norm.unsqueeze(-1).unsqueeze(-1)).to(dtype=h.dtype)
            alpha = self.alpha_net(alpha)
            z = F.softplus(alpha / self.temperature, beta=0.693, threshold=2.0)
            if self.no_mask:
                perturb = z
            else:
                perturb = torch.ones_like(z) * (1.0 - scale) + z * scale
            new_h = h * perturb
        elif self.noise_type == "learnable_additive":
            # Sample alpha from N(mu, I)
            alpha = self.alpha_net(mu + torch.randn_like(mu))
            # z = F.softplus(alpha / self.temperature, beta=0.693, threshold=2.0)
            z = alpha
            # Compute perturbation
            # scale = 1: apply perturb
            # scale = 0: do not perturb
            if noise_mask is not None:
                scale = noise_mask.unsqueeze(-1) * scale

            perturb = torch.zeros_like(z) * (1.0 - scale) + z * scale
            # Compute new_h
            new_h = h + perturb
        elif self.noise_type == "learnable":
            # Sample alpha from N(mu, I)
            if not self.no_alpha_net:
                if self.no_sampling:
                    alpha = self.alpha_net(mu)
                else:
                    alpha = self.alpha_net(mu + torch.randn_like(mu))
            else:
                if self.no_sampling:
                    alpha = mu
                else:
                    alpha = mu + torch.randn_like(mu)
            z = F.softplus(alpha / self.temperature, beta=0.693, threshold=2.0)
            # Compute perturbation
            # scale = 1: apply perturb
            # scale = 0: do not perturb
            if noise_mask is not None:
                scale = noise_mask.unsqueeze(-1) * scale

            if self.no_mask:
                perturb = z
            else:
                perturb = torch.ones_like(z) * (1.0 - scale) + z * scale
            # Compute new_h
            new_h = h * perturb
        else:
            raise NotImplementedError

        apply_perturb = apply_perturb.unsqueeze(-1).unsqueeze(-1).to(dtype=h.dtype)
        new_h = (1.0 - apply_perturb) * h + apply_perturb * new_h
        
        return new_h, scale_loss

class PerturbModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.perturb_config = config.perturb_config
        n_layers = self.perturb_config.n_layers
        self.layers = nn.ModuleList([PerturbLayer(config) for _ in range(n_layers)])