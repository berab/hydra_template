import torch 
import math
from torch import nn

class FCNN(nn.Module):
    def __init__(self, in_channels: int, in_features: int, out_features: int, 
                 depth: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = nn.ReLU()
        self.pooling = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        self.depth = depth
        self.n_leaves = 2 ** depth

        l1_init_factor = 1.0 / math.sqrt(self.in_features*in_channels)
        self.node_weights = nn.Parameter(torch.empty(((self.n_leaves-1), in_features*in_channels))
                                         .uniform_(-l1_init_factor, +l1_init_factor))
        self.node_biases = nn.Parameter(torch.empty(((self.n_leaves-1), 1))
                                        .uniform_(-l1_init_factor, +l1_init_factor))

        # CNN
        scale = 16
        self.leaf_width = 256 // self.n_leaves
        l2_init_factor = 1.0 / math.sqrt(self.leaf_width)
        self.cw1 = nn.Parameter(torch.empty((scale, in_channels, 3, 3))
                                .uniform_(-l1_init_factor, +l1_init_factor))
        self.cb1 = nn.Parameter(torch.empty((scale))
                                .uniform_(-l1_init_factor, +l1_init_factor))
        self.cw2 = nn.Parameter(torch.empty((2 * scale, scale, 3, 3))
                                .uniform_(-l2_init_factor, +l2_init_factor), requires_grad=True)
        self.cb2 = nn.Parameter(torch.empty((2 * scale))
                                .uniform_(-l1_init_factor, +l1_init_factor))

        # Classifier
        l2_init_factor = 1.0 / math.sqrt(self.leaf_width)
        self.leaf_width = 256 // self.n_leaves
        self.fw1 = nn.Parameter(torch.empty((self.n_leaves, (2 * scale // self.n_leaves) * (in_features // 16), self.leaf_width))
                                .uniform_(-l1_init_factor, +l1_init_factor))
        self.fb1 = nn.Parameter(torch.empty((self.n_leaves, self.leaf_width))
                                .uniform_(-l1_init_factor, +l1_init_factor))
        self.fw2 = nn.Parameter(torch.empty((self.n_leaves, self.leaf_width, out_features))
                                .uniform_(-l2_init_factor, +l2_init_factor))
        self.fb2 = nn.Parameter(torch.empty((self.n_leaves, out_features))
                                .uniform_(-l2_init_factor, +l2_init_factor))

    def training_forward(self, x: torch.Tensor):
        original_shape = x.shape
        x = x.view(len(x), -1)
        batch_size = x.shape[0]

        current_mixture = torch.ones((batch_size, self.n_leaves), dtype=torch.float, device=x.device)
        for current_depth in range(self.depth):
            platform = torch.tensor(2 ** current_depth - 1, dtype=torch.long, device=x.device)
            next_platform = torch.tensor(2 ** (current_depth+1) - 1, dtype=torch.long, device=x.device)
            n_nodes = 2 ** current_depth

            current_weights = self.node_weights[platform:next_platform] # ((self.n_leaves-1), in_features)    
            current_biases = self.node_biases[platform:next_platform]   # ((self.n_leaves-1), 1)

            boundary_plane_coeff_scores = torch.matmul(x, current_weights.transpose(0, 1))      # (batch_size, (self.n_leaves-1))
            boundary_plane_logits = boundary_plane_coeff_scores + current_biases.transpose(0, 1)# (batch_size, (self.n_leaves-1))
            boundary_effect = torch.sigmoid(boundary_plane_logits)                              # (batch_size, (self.n_leaves-1))

            not_boundary_effect = 1 - boundary_effect                                   # (batch_size, (self.n_leaves-1))

            mixture_modifier = torch.cat( # this cat-fu is to interleavingly combine the two tensors
                                         (not_boundary_effect.unsqueeze(-1), boundary_effect.unsqueeze(-1)),
                                         dim=-1
                                         ).flatten(start_dim=-2, end_dim=-1).unsqueeze(-1)                                               # (batch_size, (self.n_leaves-1)*2, 1)
            current_mixture = current_mixture.view(batch_size, 2 * n_nodes, self.n_leaves // (2 * n_nodes)) # (batch_size, 2*(self.n_leaves-1), self.n_leaves // (2*(self.n_leaves-1)))
            current_mixture.mul_(mixture_modifier)                                                          # (batch_size, 2*(self.n_leaves-1), self.n_leaves // (2*(self.n_leaves-1)))
            current_mixture = current_mixture.flatten(start_dim=1, end_dim=2)                               # (batch_size, self.n_leaves)

            del mixture_modifier, boundary_effect, not_boundary_effect, boundary_plane_logits, boundary_plane_coeff_scores, current_weights, current_biases

        # CNN
        x = x.view(original_shape)
        x = torch.conv2d(x, self.cw1, self.cb1, stride=1, padding=1)
        x = self.pooling(self.activation(x))
        x = torch.conv2d(x, self.cw2, self.cb2, stride=1, padding=1)
        x = self.pooling(self.activation(x))
        x = x.view(batch_size, self.n_leaves, -1)

        # Classifier
        x = torch.matmul(x.unsqueeze(-1).transpose(2, 3), 
                         self.fw1.expand(batch_size, -1, -1, -1))
        x = x.squeeze(2)
        x += self.fb1.view(1, *self.fb1.shape)                                 # (batch_size, self.n_leaves, self.leaf_width)
        x = self.activation(x)                               # (batch_size, self.n_leaves, self.leaf_width)
        out = torch.empty((batch_size, self.n_leaves, self.out_features), 
                          device=x.device)
        for i in range(self.n_leaves):
            out[:, i] = torch.matmul(
                x[:, i],
                self.fw2[i]
            ) + self.fb2[i]
        # new_logits has shape (batch_size, self.n_leaves, self.out_features)

        out *= current_mixture.unsqueeze(-1)         # (batch_size, self.n_leaves, self.out_features)
        out = out.sum(dim=1)                # (batch_size, self.out_features)
        return out

    def forward(self, x: torch.Tensor):
        if self.training:
            return self.training_forward(x)
        else:
            return self.eval_forward(x)

    def eval_forward(self, x: torch.Tensor, return_leaves: bool = False) -> torch.Tensor:
        original_shape = x.shape
        x = x.view(len(x), -1)
        batch_size = x.shape[0]

        current_nodes = torch.zeros((batch_size,), dtype=torch.long, device=x.device)
        # Routing
        for i in range(self.depth):
            plane_coeffs = self.node_weights.index_select(dim=0, index=current_nodes)       # (batch_size, in_features)
            plane_offsets = self.node_biases.index_select(dim=0, index=current_nodes)       # (batch_size, 1)
            plane_coeff_score = torch.bmm(x.unsqueeze(1), plane_coeffs.unsqueeze(-1))       # (batch_size, 1, 1)
            plane_score = plane_coeff_score.squeeze(-1) + plane_offsets                     # (batch_size, 1)
            plane_choices = (plane_score.squeeze(-1) >= 0).long()                           # (batch_size,)

            platform = torch.tensor(2 ** i - 1, dtype=torch.long, device=x.device)          # (batch_size,)
            next_platform = torch.tensor(2 ** (i+1) - 1, dtype=torch.long, device=x.device) # (batch_size,)
            current_nodes = (current_nodes - platform) * 2 + plane_choices + next_platform  # (batch_size,)

        leaves = current_nodes - next_platform              # (batch_size,) # pyright: ignore


        out = torch.empty((batch_size, self.out_features), device=x.device)
        for i in range(leaves.shape[0]):
            leaf_index = leaves[i]

            # CNN
            conv_out = x.view(original_shape)[i].unsqueeze(0)
            conv_out = torch.conv2d(conv_out, self.cw1, self.cb1, stride=1, padding=1)
            conv_out = self.pooling(self.activation(conv_out))
            conv_out = torch.conv2d(conv_out, self.cw2, self.cb2, stride=1, padding=1)
            conv_out = self.pooling(self.activation(conv_out))
            conv_out = conv_out.view(1, self.n_leaves, -1)[:, leaf_index]

            # Classifier
            conv_out = self.flatten(conv_out)
            logits = torch.matmul(
                conv_out,                  # (1, self.in_features)
                self.fw1[leaf_index]                # (self.in_features, self.leaf_width)
            )
            logits += self.fb1[leaf_index].unsqueeze(-2)    # (1, self.leaf_width)
            logits = self.activation(logits)           # (1, self.leaf_width)
            out[i] = torch.matmul(
                logits,
                self.fw2[leaf_index]
            ).squeeze(-2) + self.fb2[leaf_index]

        if return_leaves:
            return leaves.view(*original_shape[:-1]) # (..., self.out_features), (...,)
        return out

    def get_config(self):
        return {
            'leaf_width': self.leaf_width,
            'depth': self.depth,
        }
