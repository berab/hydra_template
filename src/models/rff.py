import torch 
import math
from torch import nn

class RFF(nn.Module):
    def __init__(self,
                 in_features: int, leaf_width: int, out_features: int, depth: int, n_trees: int):
        super().__init__()
        self.in_features = in_features
        self.leaf_width = leaf_width
        self.out_features = out_features
        self.activation = nn.ReLU()
        self.n_trees = n_trees
        self.n_leaves = 2 ** depth
        self.n_routers = self.n_leaves - 1

        self.depth = depth
        self.n_leaves = 2 ** depth

        l1_init_factor = 1.0 / math.sqrt(self.in_features)
        self.rw = nn.Parameter(torch.empty((self.n_trees, (self.n_leaves-1), in_features), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.rb = nn.Parameter(torch.empty((self.n_trees, (self.n_leaves-1), 1), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)

        l2_init_factor = 1.0 / math.sqrt(self.leaf_width)
        self.w1s = nn.Parameter(torch.empty((self.n_trees, self.n_leaves, in_features, leaf_width), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.b1s = nn.Parameter(torch.empty((self.n_trees, self.n_leaves, leaf_width), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.w2s = nn.Parameter(torch.empty((self.n_trees, self.n_leaves, leaf_width, out_features), dtype=torch.float).uniform_(-l2_init_factor, +l2_init_factor), requires_grad=True)
        self.b2s = nn.Parameter(torch.empty((self.n_trees, self.n_leaves, out_features), dtype=torch.float).uniform_(-l2_init_factor, +l2_init_factor), requires_grad=True)

    def training_forward(self, x: torch.Tensor, n: int):
        x = x.view(len(x), -1)
        # x has shape (batch_size, in_features)
        original_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        batch_size = x.shape[0]


        if x.shape[-1] != self.in_features:
            raise ValueError(f"input tensor must have shape (..., {self.in_features})")

        current_mixture = torch.ones((batch_size, self.n_leaves), dtype=torch.float, device=x.device)
        probs = torch.einsum("b i, r i  -> b r", x, self.rw[n]) + self.rb[n].squeeze(-1)
        probs = torch.sigmoid(probs).unsqueeze(-1)
        not_probs = 1 - probs
        for d in range(self.depth):
            platform, next_platform  = (2 ** d - 1), (2 ** (d+1) - 1)
            cur_probs = probs[:, platform:next_platform]
            n_nodes = 2 ** d

            mixture_modifier = torch.cat( # this cat-fu is to interleavingly combine the two tensors
                                         (1-cur_probs, cur_probs),
                                         dim=-1
                                         ).flatten(start_dim=-2, end_dim=-1).unsqueeze(-1)                                               # (batch_size, (self.n_leaves-1)*2, 1)
            current_mixture = current_mixture.view(batch_size, 2 * n_nodes, self.n_leaves // (2 * n_nodes)) # (batch_size, 2*(self.n_leaves-1), self.n_leaves // (2*(self.n_leaves-1)))
            current_mixture.mul_(mixture_modifier)                                                          # (batch_size, 2*(self.n_leaves-1), self.n_leaves // (2*(self.n_leaves-1)))
            current_mixture = current_mixture.flatten(start_dim=1, end_dim=2)                               # (batch_size, self.n_leaves)

        element_logits = torch.matmul(x, self.w1s[n].transpose(0, 1).flatten(1, 2))            # (batch_size, self.n_leaves * self.leaf_width)
        element_logits = element_logits.view(batch_size, self.n_leaves, self.leaf_width)    # (batch_size, self.n_leaves, self.leaf_width)
        element_logits += self.b1s[n].view(1, *self.b1s[n].shape)                                 # (batch_size, self.n_leaves, self.leaf_width)
        element_activations = self.activation(element_logits)                               # (batch_size, self.n_leaves, self.leaf_width)
        new_logits = torch.einsum("b l h, l h o->b l o", element_activations, self.w2s[n]) + self.b2s[n]
        new_logits *= current_mixture.unsqueeze(-1)         # (batch_size, self.n_leaves, self.out_features)
        final_logits = new_logits.sum(dim=1)                # (batch_size, self.out_features)

        final_logits = final_logits.view(*original_shape[:-1], self.out_features)   # (..., self.out_features)
        return final_logits, current_mixture

    def forward(self, x: torch.Tensor, n: int = 0):
        if self.training:
            return self.training_forward(x, n)
        else:
            return self.eval_forward(x)

    def eval_forward(self, x: torch.Tensor, return_leaves: bool = False) -> torch.Tensor:
        x = x.view(len(x), -1)
        original_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        batch_size = x.shape[0]
        # x has shape (batch_size, in_features)

        all_out_logits = torch.empty((batch_size, self.n_trees, self.out_features), dtype=torch.float, device=x.device)
        all_leaves = torch.empty((batch_size, self.n_trees), dtype=torch.long, device=x.device)
        for n in range(self.n_trees):
            current_nodes = torch.zeros((batch_size,), dtype=torch.long, device=x.device)
            for i in range(self.depth):
                plane_coeffs = self.rw[n].index_select(dim=0, index=current_nodes)       # (batch_size, in_features)
                plane_offsets = self.rb[n].index_select(dim=0, index=current_nodes)       # (batch_size, 1)
                plane_coeff_score = torch.bmm(x.unsqueeze(1), plane_coeffs.unsqueeze(-1))       # (batch_size, 1, 1)
                plane_score = plane_coeff_score.squeeze(-1) + plane_offsets                     # (batch_size, 1)
                plane_choices = (plane_score.squeeze(-1) >= 0).long()                           # (batch_size,)

                platform = torch.tensor(2 ** i - 1, dtype=torch.long, device=x.device)          # (batch_size,)
                next_platform = torch.tensor(2 ** (i+1) - 1, dtype=torch.long, device=x.device) # (batch_size,)
                current_nodes = (current_nodes - platform) * 2 + plane_choices + next_platform  # (batch_size,)

            leaves = current_nodes - next_platform              # (batch_size,)

            l1_weights = self.w1s[n].index_select(dim=0, index=leaves) # b, i, w
            b1_weights = self.b1s[n].index_select(dim=0, index=leaves) # b, w
            l2_weights = self.w2s[n].index_select(dim=0, index=leaves) # b, w, o
            b2_weights = self.b2s[n].index_select(dim=0, index=leaves) # b, w
            logits = torch.einsum("b i, b i w -> b w", x, l1_weights) + b1_weights
            logits = self.activation(logits)
            logits = torch.einsum("b w, b w o -> b o", logits, l2_weights) + b2_weights

            out_logits, leaves = logits.view(*original_shape[:-1], self.out_features), leaves.view(*original_shape[:-1]) # (..., self.out_features), (...,)
            all_out_logits[:, n], all_leaves[:, n] = out_logits, leaves

        if return_leaves:
            return all_leaves
        return all_out_logits

    def get_config(self):
        return {
            'leaf_width': self.leaf_width,
            'depth': self.depth,
            'n_trees': self.n_trees,
            'complexity': (self.depth + self.leaf_width) * self.n_trees,
            'size': (self.n_leaves * self.leaf_width + self.n_routers) * self.n_trees
        }
