"""
Scheduler for Continuous Batching in PagedAttention v3.

Manages sequence states and schedules prefill/decode steps for multiple sequences.
"""

from enum import Enum
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


class SequenceState(Enum):
    """State of a sequence in the scheduler."""
    PREFILL = "prefill"      # Processing prompt tokens
    DECODE = "decode"         # Generating tokens
    FINISHED = "finished"     # Generation complete or stopped


@dataclass
class SequenceInfo:
    """Information about a sequence in the batch."""
    seq_id: int
    state: SequenceState
    prompt_tokens: List[int]
    generated_tokens: List[int]
    total_tokens: int
    position: int  # Current position in sequence
    max_new_tokens: int
    finished: bool = False
    
    def get_current_token(self) -> Optional[int]:
        """Get the current token to process."""
        if self.state == SequenceState.PREFILL:
            if self.position < len(self.prompt_tokens):
                return self.prompt_tokens[self.position]
        elif self.state == SequenceState.DECODE:
            if self.generated_tokens:
                return self.generated_tokens[-1]
            elif self.prompt_tokens:
                return self.prompt_tokens[-1]
        return None
    
    def advance(self):
        """Advance to next position."""
        if self.state == SequenceState.PREFILL:
            self.position += 1
            if self.position >= len(self.prompt_tokens):
                # Prefill complete, move to decode
                self.state = SequenceState.DECODE
                self.position = len(self.prompt_tokens)
        elif self.state == SequenceState.DECODE:
            self.position += 1
            if len(self.generated_tokens) >= self.max_new_tokens:
                self.finished = True
                self.state = SequenceState.FINISHED


class ContinuousBatchScheduler:
    """
    Scheduler for continuous batching.
    
    Manages multiple sequences concurrently, scheduling prefill and decode steps.
    """
    
    def __init__(self, max_batch_size: int = 32):
        """
        Initialize the scheduler.
        
        Args:
            max_batch_size: Maximum number of sequences to process in one batch
        """
        self.max_batch_size = max_batch_size
        self.sequences: Dict[int, SequenceInfo] = {}
        self.next_seq_id = 0
    
    def add_sequence(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 50,
        immediate_prefill: bool = False
    ) -> int:
        """
        Add a new sequence to the scheduler.
        
        Args:
            prompt_tokens: Tokenized prompt
            max_new_tokens: Maximum tokens to generate
            immediate_prefill: If True, sequence is ready for prefill immediately.
                              If False, stays in PREFILL state for batching.
            
        Returns:
            Sequence ID
        """
        seq_id = self.next_seq_id
        self.next_seq_id += 1
        
        self.sequences[seq_id] = SequenceInfo(
            seq_id=seq_id,
            state=SequenceState.PREFILL,
            prompt_tokens=prompt_tokens,
            generated_tokens=[],
            total_tokens=len(prompt_tokens),
            position=0,
            max_new_tokens=max_new_tokens,
            finished=False
        )
        
        return seq_id
    
    def get_prefill_batch(self, max_batch_size: Optional[int] = None) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        Get sequences ready for prefill batching.
        
        Returns sequences in PREFILL state that haven't started processing yet.
        
        Args:
            max_batch_size: Maximum number of sequences to include (None = all available)
            
        Returns:
            Tuple of (seq_ids, prompt_token_lists, positions_start, total_tokens)
            where positions_start indicates where each sequence starts in the flattened array
        """
        prefill_seqs = []
        
        for seq_id, seq_info in self.sequences.items():
            if seq_info.state == SequenceState.PREFILL and seq_info.position == 0:
                prefill_seqs.append(seq_info)
                if max_batch_size and len(prefill_seqs) >= max_batch_size:
                    break
        
        if not prefill_seqs:
            return [], [], [], []
        
        seq_ids = [s.seq_id for s in prefill_seqs]
        prompt_token_lists = [s.prompt_tokens for s in prefill_seqs]
        
        # Calculate positions_start: where each sequence starts in flattened array
        positions_start = []
        total_tokens = 0
        for s in prefill_seqs:
            positions_start.append(total_tokens)
            total_tokens += len(s.prompt_tokens)
        
        return seq_ids, prompt_token_lists, positions_start, [len(s.prompt_tokens) for s in prefill_seqs]
    
    def get_batch(
        self,
        include_prefill: bool = True,
        include_decode: bool = True
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Get sequences ready for processing in current step.
        
        Args:
            include_prefill: Include sequences in prefill state
            include_decode: Include sequences in decode state
            
        Returns:
            Tuple of (seq_ids, positions, token_ids) for current batch
        """
        seq_ids = []
        positions = []
        token_ids = []
        
        for seq_id, seq_info in self.sequences.items():
            if seq_info.finished:
                continue
            
            if seq_info.state == SequenceState.PREFILL and include_prefill:
                if seq_info.position < len(seq_info.prompt_tokens):
                    seq_ids.append(seq_id)
                    positions.append(seq_info.position)
                    token_ids.append(seq_info.prompt_tokens[seq_info.position])
            
            elif seq_info.state == SequenceState.DECODE and include_decode:
                current_token = seq_info.get_current_token()
                if current_token is not None:
                    seq_ids.append(seq_id)
                    positions.append(seq_info.position)
                    token_ids.append(current_token)
        
        return seq_ids, positions, token_ids
    
    def update_sequences(
        self,
        seq_ids: List[int],
        next_token_ids: List[int],
        eos_token_id: Optional[int] = None
    ):
        """
        Update sequences after processing a batch step.
        
        Args:
            seq_ids: Sequence IDs that were processed
            next_token_ids: Generated next token IDs for each sequence
            eos_token_id: EOS token ID (if provided, sequences will finish on EOS)
        """
        for seq_id, next_token_id in zip(seq_ids, next_token_ids):
            if seq_id not in self.sequences:
                continue
            
            seq_info = self.sequences[seq_id]
            
            # Check for EOS
            if eos_token_id is not None and next_token_id == eos_token_id:
                seq_info.finished = True
                seq_info.state = SequenceState.FINISHED
                continue
            
            # Update sequence
            if seq_info.state == SequenceState.PREFILL:
                # Prefill: advance position
                seq_info.advance()
            elif seq_info.state == SequenceState.DECODE:
                # Decode: add generated token and advance
                seq_info.generated_tokens.append(next_token_id)
                seq_info.total_tokens += 1
                seq_info.advance()
    
    def remove_sequence(self, seq_id: int):
        """Remove a finished sequence from the scheduler."""
        if seq_id in self.sequences:
            del self.sequences[seq_id]
    
    def get_finished_sequences(self) -> List[int]:
        """Get list of finished sequence IDs."""
        return [
            seq_id for seq_id, seq_info in self.sequences.items()
            if seq_info.finished
        ]
    
    def get_active_count(self) -> int:
        """Get number of active (non-finished) sequences."""
        return sum(1 for s in self.sequences.values() if not s.finished)
    
    def get_stats(self) -> Dict:
        """Get scheduler statistics."""
        prefill_count = sum(
            1 for s in self.sequences.values()
            if s.state == SequenceState.PREFILL
        )
        decode_count = sum(
            1 for s in self.sequences.values()
            if s.state == SequenceState.DECODE
        )
        finished_count = sum(
            1 for s in self.sequences.values()
            if s.finished
        )
        
        return {
            "total_sequences": len(self.sequences),
            "prefill": prefill_count,
            "decode": decode_count,
            "finished": finished_count,
            "active": self.get_active_count()
        }
