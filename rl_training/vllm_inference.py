"""
vLLM Inference Wrapper for RL Training.

Provides a clean interface for generating rollouts using vLLM.
"""

from typing import List, Optional, Tuple

import torch
from vllm import LLM, SamplingParams


class VLLMInferenceEngine:
    """
    Wrapper around vLLM for efficient batched inference during RL training.
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
        dtype: str = "auto",
        runner: str = "generate",
    ):
        """
        Initialize vLLM inference engine.

        Args:
            model_path: Path to model weights
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_model_len: Maximum sequence length
            dtype: Data type for model weights
            runner: Runner type ("generate" for v0, "generate_v1" for v1)
        """
        self.model_path = model_path
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            dtype=dtype,
            runner=runner,
        )

    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        logprobs: int = 1,
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Generate completions for a batch of prompts.

        Args:
            prompts: List of prompt strings
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            logprobs: Number of log probabilities to return

        Returns:
            List of (completion_text, log_probs_tensor) tuples
        """
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )

        outputs = self.llm.generate(prompts, sampling_params)

        results = []
        for output in outputs:
            completion = output.outputs[0].text

            # Extract log probabilities
            log_probs_list = []
            for token_logprobs in output.outputs[0].logprobs:
                if token_logprobs:
                    # Get the log prob of the sampled token
                    sampled_token_id = list(token_logprobs.keys())[0]
                    log_prob = token_logprobs[sampled_token_id].logprob
                    log_probs_list.append(log_prob)

            log_probs_tensor = torch.tensor(log_probs_list)
            results.append((completion, log_probs_tensor))

        return results

    def generate_with_logits(
        self,
        prompts: List[str],
        max_tokens: int = 128,
        temperature: float = 1.0,
    ) -> List[Tuple[str, List[int], torch.Tensor]]:
        """
        Generate completions and return full logits for each position.

        This is useful for computing log probs under different policies.

        Args:
            prompts: List of prompt strings
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature

        Returns:
            List of (completion_text, token_ids, logits_tensor) tuples
            logits_tensor shape: (seq_len, vocab_size)
        """
        # Note: This requires accessing vLLM internals or using a custom
        # sampling method. For now, we'll use the standard generate method.
        # In production, you may need to modify vLLM to expose logits.

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=1,
        )

        outputs = self.llm.generate(prompts, sampling_params)

        results = []
        for output in outputs:
            completion = output.outputs[0].text
            token_ids = output.outputs[0].token_ids

            # For now, we can't get full logits easily from vLLM
            # This would require modifications to vLLM's output format
            # Placeholder: return empty tensor
            logits = torch.empty(0)  # TODO: Implement logits extraction

            results.append((completion, token_ids, logits))

        return results

    def update_weights(self, state_dict: dict):
        """
        Update model weights from TorchTitan training.

        Args:
            state_dict: State dictionary with updated weights
        """
        # This requires access to vLLM's model internals
        # The implementation depends on the TorchTitan adapter
        if hasattr(self.llm, 'llm_engine') and hasattr(self.llm.llm_engine, 'model_executor'):
            model_executor = self.llm.llm_engine.model_executor
            if hasattr(model_executor, 'load_state_dict'):
                model_executor.load_state_dict(state_dict)
            else:
                # Access the actual model
                model = model_executor.driver_worker.model_runner.model
                model.load_state_dict(state_dict, strict=False)
        else:
            raise NotImplementedError("Weight updating requires vLLM internal access")

    def get_weights(self) -> dict:
        """
        Extract current model weights for training.

        Returns:
            State dictionary of model weights
        """
        if hasattr(self.llm, 'llm_engine') and hasattr(self.llm.llm_engine, 'model_executor'):
            model_executor = self.llm.llm_engine.model_executor
            model = model_executor.driver_worker.model_runner.model
            return model.state_dict()
        else:
            raise NotImplementedError("Weight extraction requires vLLM internal access")


class RolloutBuffer:
    """
    Buffer for storing rollout data during RL training.
    """

    def __init__(self):
        self.prompts = []
        self.completions = []
        self.log_probs = []
        self.rewards = []

    def add(
        self,
        prompt: str,
        completion: str,
        log_probs: torch.Tensor,
        reward: float,
    ):
        """Add a single rollout sample."""
        self.prompts.append(prompt)
        self.completions.append(completion)
        self.log_probs.append(log_probs)
        self.rewards.append(reward)

    def add_batch(
        self,
        prompts: List[str],
        completions: List[str],
        log_probs: List[torch.Tensor],
        rewards: List[float],
    ):
        """Add a batch of rollout samples."""
        self.prompts.extend(prompts)
        self.completions.extend(completions)
        self.log_probs.extend(log_probs)
        self.rewards.extend(rewards)

    def get_batch(self):
        """Get all stored samples as a batch."""
        from .grpo import RLBatch

        return RLBatch(
            prompts=self.prompts,
            responses=self.completions,
            rewards=torch.tensor(self.rewards),
            log_probs=self.log_probs,
        )

    def clear(self):
        """Clear the buffer."""
        self.prompts.clear()
        self.completions.clear()
        self.log_probs.clear()
        self.rewards.clear()

    def __len__(self):
        return len(self.prompts)
