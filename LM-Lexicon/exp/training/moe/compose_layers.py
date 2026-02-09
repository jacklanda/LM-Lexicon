from typing import Optional

import math
import torch
import torch.nn.functional as F
from torch import nn


def convert_linear_to_moe(
    name: str,
    config: dict,
    layer_idx: int,
    in_features: int,
    out_features: int,
    bias: bool = True,
    routing_policy: Optional[str] = None,
):
    """Converts nn.Linear to MoeLayer
    Args:
        name (str): Layer Name
        config (dict): Composer config
        layer_idx (int): Transformer block id.
        in_features (int): Input features of Default nn.Linear layer.
        out_features (int): Output features of Default nn.Linear layer.
        bias (bool, optional): Defaults to True.
        routing_policy (Optional[str], optional): Routing Policy. Defaults to None.
    """
    if (layer_idx in config.router_layers_index) and (name in config.router_layers):
        return MoeLayer(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            routing_policy=routing_policy,
        )
    return nn.Linear(in_features, out_features, bias=bias)


class MoeLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        num_experts: int,
        num_experts_per_tok: int = 2,
        routing_policy: Optional[str] = None,
    ):
        """Mixture of Expert Layer

        Args:
            in_features (int): Input Features
            out_features (int): Output Features
            bias (bool): bias
            num_experts (int): Total numbers of experts that Router Layer would handle
            num_experts_per_tok (int, optional): Number of Active Experts per token(step). Defaults to 2.
            routing_policy (Optional[str], optional): Routing Policy. Defaults to None.
        """
        super().__init__()
        self.routing_policy = routing_policy
        if routing_policy == "token-level":
            # topk token-level routing
            self.gate = nn.Linear(in_features, num_experts, bias=False)
            self.experts = nn.ModuleList(
                [nn.Linear(in_features, out_features, bias) for _ in range(num_experts)]
            )
            self.num_experts_per_tok = num_experts_per_tok
            self.in_features = in_features
            self.out_features = out_features
        elif routing_policy in ["soft-sequence-level", "hard-sequence-level"]:
            # sequence-level routing
            self.gate = nn.Linear(in_features, num_experts, bias=False)
            self.num_experts = num_experts
            self.experts = nn.ModuleList(
                [nn.Linear(in_features, out_features) for _ in range(num_experts)]
            )
        elif routing_policy == "domain-level":
            # domain-level routing
            self.gate = nn.Linear(in_features, num_experts, bias=False)
            self.num_experts = num_experts
            self.experts = nn.ModuleList(
                [nn.Linear(in_features, out_features) for _ in range(num_experts)]
            )
        else:
            raise ValueError(f"Invalid routing policy: {routing_policy}")

    def forward(
        self, inputs: torch.Tensor, domain_labels: Optional[torch.Tensor] = None
    ):
        # shape of inputs: (batch_size, seq_len, in_features)
        if not self.gate.weight.is_meta and inputs.device != self.gate.weight.device:
            # only move inputs if weight of gate is not meta and inputs are on different device
            # print(f"Moving inputs ({inputs.device}) to device ({self.device})")
            # inputs = inputs.to(self.gate.weight.device)
            inputs = inputs.to(self.experts[0].weight.device)

        if self.routing_policy == "token-level":
            gate_logits = self.gate(inputs)
            weights, selected_experts = torch.topk(
                gate_logits, self.num_experts_per_tok
            )
            weights = F.softmax(weights, dim=2, dtype=torch.float).to(inputs.dtype)
            results = torch.zeros(
                (inputs.shape[0], inputs.shape[1], self.out_features),
                device=inputs.device,
                dtype=inputs.dtype,
            )
            weights = weights.to(inputs.device)
            for ix, expert in enumerate(self.experts):
                batch_idx, tok_idx, expert_idx = torch.where(selected_experts == ix)
                results[batch_idx, tok_idx] += expert(
                    inputs[batch_idx, tok_idx]
                ) * weights[batch_idx, tok_idx, expert_idx].unsqueeze(-1)
        elif self.routing_policy == "soft-sequence-level":
            # soft sequence-level routing (each expert is responsible for the entire sequence)
            gate_logits = self.gate(inputs)  # (batch_size, seq_len, num_experts)
            # TODO: considering replace "mean" to "sum" over all tokens in a sequence
            gate_logits_mean = gate_logits.mean(dim=1)  # (batch_size, num_experts)
            weights = F.softmax(gate_logits_mean, dim=-1)  # (batch_size, num_experts)
            results = torch.zeros(
                (inputs.shape[0], inputs.shape[1], self.out_features),
                device=inputs.device,
                dtype=inputs.dtype,
            )
            for ix, expert in enumerate(self.experts):
                results += expert(inputs) * weights[:, ix].unsqueeze(-1)
        elif self.routing_policy == "hard-sequence-level":
            # hard sequence-level routing (only one selected expert is responsible for the entire sequence)
            gate_logits = self.gate(inputs)  # (batch_size, seq_len, num_experts)
            # TODO: considering replace "mean" to "sum" over all tokens in a sequence
            gate_logits_mean = gate_logits.mean(dim=1)  # (batch_size, num_experts)
            _, selected_experts = torch.topk(gate_logits_mean, 1)  # (batch_size, 1)
            results = torch.zeros(
                (inputs.shape[0], inputs.shape[1], self.out_features),
                device=inputs.device,
                dtype=inputs.dtype,
            )
            for ix, expert in enumerate(self.experts):
                results += expert(inputs) * (selected_experts == ix).float().unsqueeze(
                    -1
                )
        elif self.routing_policy == "domain-level":
            # domain-level routing (only one selected expert is responsible for the entire sequence)
            gate_logits = self.gate(inputs)  # (batch_size, seq_len, num_experts)
            # select the only one expert by domain_labels (domain_labels is a tensor of shape (batch_size, 1))
            results = torch.zeros(
                (inputs.shape[0], inputs.shape[1], self.out_features),
                device=inputs.device,
                dtype=inputs.dtype,
            )
            for ix, expert in enumerate(self.experts):
                results += expert(inputs) * (domain_labels == ix).float().unsqueeze(-1)

        return results
