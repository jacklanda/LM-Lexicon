import torch
import torch.nn as nn
import torch.nn.functional as F


class MoeLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        num_experts: int,
        num_experts_per_tok: int = 2,
    ):
        """Mixture of Expert Layer
        Args:
            in_features (int): Input Features
            out_features (int): Output Features
            bias (bool): bias
            num_experts (int): Total numbers of experts that Router Layer would handle
            num_experts_per_tok (int, optional): Number of Active Experts per token(step). Defaults to 2.
        """
        super().__init__()
        self.gate = nn.Linear(in_features, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [nn.Linear(in_features, out_features, bias) for _ in range(num_experts)]
        )
        self.num_experts_per_tok = num_experts_per_tok
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, inputs: torch.Tensor):
        gate_logits = self.gate(inputs)
        weights, selected_experts = torch.topk(
            gate_logits, self.num_experts_per_tok, dim=-1
        )
        weights = F.softmax(weights, dim=-1, dtype=torch.float).to(inputs.dtype)

        # Initialize a tensor to hold the results
        results = torch.zeros(
            (inputs.shape[0], inputs.shape[1], self.out_features),
            device=inputs.device,
            dtype=inputs.dtype,
        )

        # Flatten the batch and sequence dimensions for easier manipulation
        batch_size, seq_len, _ = inputs.size()
        flat_inputs = inputs.view(-1, self.in_features)

        # Iterate over each expert and compute its contribution in a batched manner
        for ix, expert in enumerate(self.experts):
            # Create a mask for the current expert
            mask = selected_experts == ix
            if mask.any():
                # Flatten the mask to match the flattened inputs dimension
                mask_expanded = mask.view(-1)

                # Gather inputs for the current expert
                expert_inputs = flat_inputs[mask_expanded]

                # Compute the expert's output
                expert_outputs = expert(expert_inputs)

                # Gather corresponding weights and expand them to match expert_outputs
                expert_weights = weights.view(-1, self.num_experts_per_tok)[
                    mask_expanded
                ]
                expert_weights = expert_weights.unsqueeze(-1)

                # Multiply the expert outputs by the corresponding weights
                expert_outputs_weighted = expert_outputs * expert_weights

                # Accumulate the results
                results.view(-1, self.out_features)[mask_expanded] += (
                    expert_outputs_weighted
                )

        return results


# Example usage:
# Initialize the MoeLayer
moe_layer = MoeLayer(
    in_features=10, out_features=10, bias=True, num_experts=4, num_experts_per_tok=2
)

# Create a sample input tensor
inputs = torch.randn(
    2, 3, 10
)  # Batch size of 2, sequence length of 3, input features of 10

# Forward pass
output = moe_layer(inputs)
print(output)
