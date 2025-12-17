"""
Simple Scheduler for TP RadixAttention.

This is a simplified scheduler that coordinates TP workers for inference.
In a full SGLang implementation, the scheduler is much more sophisticated,
handling batching, scheduling policies, etc.
"""
import torch
import torch.distributed as dist
from typing import List, Optional, Dict
from dataclasses import dataclass

try:
    from .parallel_state import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
except ImportError:
    from parallel_state import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )


@dataclass
class Request:
    """A single inference request."""
    request_id: int
    prompt: str
    max_new_tokens: int = 10
    temperature: float = 1.0
    top_p: float = 1.0


class TPScheduler:
    """
    Simple scheduler for coordinating TP workers.
    
    This is a simplified version of SGLang's scheduler that:
    - Manages request queue
    - Coordinates TP workers
    - Handles batching (simplified - one request at a time for now)
    """
    
    def __init__(self, model_wrapper):
        """
        Initialize the TP scheduler.
        
        Args:
            model_wrapper: TPRadixAttentionModelWrapper instance
        """
        self.model_wrapper = model_wrapper
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.is_main_rank = (self.tp_rank == 0)
        
        # Request queue (simplified - single request processing)
        self.request_queue: List[Request] = []
        self.next_request_id = 0
    
    def add_request(self, prompt: str, max_new_tokens: int = 10, **kwargs) -> int:
        """
        Add a request to the queue.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Request ID
        """
        request_id = self.next_request_id
        self.next_request_id += 1
        
        request = Request(
            request_id=request_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        
        self.request_queue.append(request)
        return request_id
    
    def process_requests(self, num_requests: Optional[int] = None) -> List[str]:
        """
        Process requests from the queue.
        
        Args:
            num_requests: Number of requests to process (None = all)
            
        Returns:
            List of generated texts
        """
        if num_requests is None:
            num_requests = len(self.request_queue)
        
        results = []
        
        for i in range(min(num_requests, len(self.request_queue))):
            request = self.request_queue[i]
            
            if self.is_main_rank:
                print(f"\n[Scheduler] Processing request {request.request_id}: {request.prompt[:50]}...")
            
            # Generate with TP
            generated_text = self.model_wrapper.generate(
                request.prompt,
                max_new_tokens=request.max_new_tokens
            )
            
            results.append(generated_text)
            
            if self.is_main_rank:
                print(f"[Scheduler] Request {request.request_id} completed")
        
        # Remove processed requests
        self.request_queue = self.request_queue[num_requests:]
        
        return results
    
    def clear_queue(self):
        """Clear the request queue."""
        self.request_queue = []
