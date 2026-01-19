from dataclasses import dataclass, field
from typing import Any, Iterator
import numpy as np

@dataclass
class Classification:
    class_id: np.ndarray
    confidence: np.ndarray | None = None
    
    def __len__(self) -> int:
        """
        Returns the number of classifications.
        """
        return len(self.class_id)
    
    @classmethod
    def from_mobilenet(cls, mobilenet_results, top_k: int | None = None):
        results = mobilenet_results[0]
        meta = mobilenet_results[1]

        # Unwrap container outputs
        if isinstance(results, (list, tuple)):
            results = results[0]

        # Convert to numpy
        if hasattr(results, "detach"):  # PyTorch tensor
            logits = results.detach().cpu().numpy()
        else:
            logits = np.asarray(results)

        # Normalize shape -> (B, C)
        logits = np.squeeze(logits)
        if logits.ndim == 1:
            logits = logits[None, :]
        elif logits.ndim != 2:
            raise ValueError(f"Invalid MobileNet output shape: {logits.shape}")

        # Logits -> probabilities
        if logits.max() > 1.0:
            exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp / np.sum(exp, axis=1, keepdims=True)
        else:
            probs = logits

        B, C = probs.shape

        # ALL classes (default)
        if top_k is None:
            class_id = np.tile(np.arange(C), (B, 1))
            confidence = probs

        # Top-1
        elif top_k == 1:
            class_id = np.argmax(probs, axis=1)
            confidence = probs[np.arange(B), class_id]

        # Top-K
        else:
            if top_k > C:
                raise ValueError("top_k cannot be greater than number of classes")

            idx = np.argpartition(-probs, top_k - 1, axis=1)[:, :top_k]
            scores = np.take_along_axis(probs, idx, axis=1)

            order = np.argsort(-scores, axis=1)
            class_id = np.take_along_axis(idx, order, axis=1)
            confidence = np.take_along_axis(scores, order, axis=1)

        return cls(
            class_id=class_id,
            confidence=confidence,
        )
        
    @classmethod
    def from_resnet(cls, resnet_results, top_k: int | None = None):
        results = resnet_results[0]
        meta = resnet_results[1]

        # Unwrap container outputs
        if isinstance(results, (list, tuple)):
            results = results[0]

        # Convert to numpy
        if hasattr(results, "detach"):  # PyTorch tensor
            logits = results.detach().cpu().numpy()
        else:
            logits = np.asarray(results)

        # Normalize shape -> (B, C)
        logits = np.squeeze(logits)
        if logits.ndim == 1:
            logits = logits[None, :]
        elif logits.ndim != 2:
            raise ValueError(f"Invalid Resnet output shape: {logits.shape}")

        # Logits -> probabilities
        if logits.max() > 1.0:
            exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp / np.sum(exp, axis=1, keepdims=True)
        else:
            probs = logits

        B, C = probs.shape

        # ALL classes (default)
        if top_k is None:
            class_id = np.tile(np.arange(C), (B, 1))
            confidence = probs

        # Top-1
        elif top_k == 1:
            class_id = np.argmax(probs, axis=1)
            confidence = probs[np.arange(B), class_id]

        # Top-K
        else:
            if top_k > C:
                raise ValueError("top_k cannot be greater than number of classes")

            idx = np.argpartition(-probs, top_k - 1, axis=1)[:, :top_k]
            scores = np.take_along_axis(probs, idx, axis=1)

            order = np.argsort(-scores, axis=1)
            class_id = np.take_along_axis(idx, order, axis=1)
            confidence = np.take_along_axis(scores, order, axis=1)

        return cls(
            class_id=class_id,
            confidence=confidence,
        )
    