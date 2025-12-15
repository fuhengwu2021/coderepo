"""
Expert Parallelism for Mixture-of-Experts (MoE)
Demonstrates how to apply EP to MoE layers

Key concepts:
1. Each GPU holds different experts
2. Tokens are dispatched to the correct expert GPU using all-to-all
3. Experts compute on their assigned tokens
4. Results are combined back using all-to-all
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try relative import first (when used as package), fallback to absolute (when used as script)
try:
    from .parallel_state import (
        get_expert_parallel_rank,
        get_expert_parallel_world_size,
        expert_parallel_all_gather,
        expert_parallel_reduce_scatter,
    )
except ImportError:
    from parallel_state import (
        get_expert_parallel_rank,
        get_expert_parallel_world_size,
        expert_parallel_all_gather,
        expert_parallel_reduce_scatter,
    )


def divide(numerator: int, denominator: int) -> int:
    """Divide numerator by denominator, ensuring integer result"""
    assert numerator % denominator == 0, f"{numerator} is not divisible by {denominator}"
    return numerator // denominator


class Expert(nn.Module):
    """A single expert MLP"""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: SiLU(gate) * up -> down"""
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        activated = F.silu(gate) * up
        return self.down_proj(activated)


class Router(nn.Module):
    """Router that selects which experts to use for each token"""
    
    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
    
    def forward(self, x: torch.Tensor, top_k: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute router logits and select top-k experts
        
        Returns:
            topk_weights: [num_tokens, top_k] - weights for selected experts
            topk_ids: [num_tokens, top_k] - expert IDs (global, not local)
        """
        router_logits = self.gate(x)  # [num_tokens, num_experts]
        topk_weights, topk_ids = torch.topk(router_logits, top_k, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1)
        return topk_weights, topk_ids


class ExpertParallelMoE(nn.Module):
    """
    Mixture-of-Experts layer with Expert Parallelism
    
    This demonstrates the key concepts:
    1. Each GPU holds a subset of experts
    2. Tokens are dispatched to expert GPUs using all-gather (dispatch)
    3. Each GPU computes on tokens assigned to its experts
    4. Results are combined using reduce-scatter (combine)
    
    Args:
        num_experts: Total number of experts
        hidden_size: Hidden dimension size
        intermediate_size: Intermediate dimension size
        top_k: Number of experts to use per token
    """
    
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        top_k: int = 2,
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        
        # Get expert parallel info
        self.ep_rank = get_expert_parallel_rank()
        self.ep_size = get_expert_parallel_world_size()
        
        # Calculate which experts this rank owns
        self.num_local_experts = divide(num_experts, self.ep_size)
        self.expert_start_idx = self.ep_rank * self.num_local_experts
        self.expert_end_idx = self.expert_start_idx + self.num_local_experts
        
        # Router is replicated on all ranks
        self.router = Router(hidden_size, num_experts)
        
        # Each rank only holds its local experts
        self.experts = nn.ModuleList([
            Expert(hidden_size, intermediate_size)
            for _ in range(self.num_local_experts)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with expert parallelism
        
        Args:
            x: Input tensor of shape [num_tokens, hidden_size]
        
        Returns:
            Output tensor of shape [num_tokens, hidden_size]
        """
        num_tokens = x.shape[0]
        
        # Step 1: Compute router logits (replicated on all ranks)
        topk_weights, topk_ids = self.router(x, top_k=self.top_k)
        # topk_ids: [num_tokens, top_k] with values in [0, num_experts-1]
        
        # Step 2: Dispatch - all-gather tokens and router info
        # In real vLLM, this uses all-to-all to route tokens to expert GPUs
        # For this simplified demo, we use all-gather so all ranks have all tokens
        x_dispatched = expert_parallel_all_gather(x, dim=0)  # [num_tokens * ep_size, hidden_size]
        topk_weights_dispatched = expert_parallel_all_gather(topk_weights, dim=0)
        topk_ids_dispatched = expert_parallel_all_gather(topk_ids, dim=0)
        
        # Step 3: Compute expert outputs for tokens assigned to this rank's experts
        # Each rank processes tokens that need its local experts
        output_list = []
        
        for token_idx in range(x_dispatched.shape[0]):
            token = x_dispatched[token_idx:token_idx+1]  # [1, hidden_size]
            token_weights = topk_weights_dispatched[token_idx]  # [top_k]
            token_expert_ids = topk_ids_dispatched[token_idx]  # [top_k]
            
            token_output = torch.zeros_like(token)
            
            # Process each selected expert for this token
            for k in range(self.top_k):
                expert_id = token_expert_ids[k].item()
                weight = token_weights[k].item()
                
                # Check if this expert is on this rank
                if self.expert_start_idx <= expert_id < self.expert_end_idx:
                    local_expert_idx = expert_id - self.expert_start_idx
                    expert_output = self.experts[local_expert_idx](token)
                    token_output += weight * expert_output
                # Note: In real implementation, tokens would only be sent to ranks
                # that have the needed experts, not all ranks
            
            output_list.append(token_output)
        
        # Combine outputs: [num_tokens * ep_size, hidden_size]
        expert_outputs = torch.cat(output_list, dim=0)
        
        # Step 4: Combine - reduce-scatter to get final results
        # In real vLLM, this uses all-to-all to route results back
        # For this demo, we use reduce-scatter to sum results from all ranks
        output = expert_parallel_reduce_scatter(expert_outputs, dim=0)
        
        # Trim to original number of tokens
        output = output[:num_tokens]
        
        return output
    
    def extra_repr(self) -> str:
        return (
            f"num_experts={self.num_experts}, "
            f"num_local_experts={self.num_local_experts}, "
            f"expert_range=[{self.expert_start_idx}, {self.expert_end_idx}), "
            f"ep_size={self.ep_size}, "
            f"top_k={self.top_k}"
        )


def create_moe(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int = 2,
) -> ExpertParallelMoE:
    """Factory function to create an expert parallel MoE layer"""
    return ExpertParallelMoE(num_experts, hidden_size, intermediate_size, top_k)

