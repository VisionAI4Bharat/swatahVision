from dataclasses import dataclass
from typing import Iterator
import numpy as np


@dataclass
class Pose:
    keypoints: np.ndarray        # (K,2) → x,y
    confidence: np.ndarray       # (K,)
    
    def __len__(self) -> int:
        """Number of keypoints"""
        return len(self.keypoints)

    def __iter__(self) -> Iterator:
        for kp, conf in zip(self.keypoints, self.confidence):
            yield kp, conf

    #  MoveNet parser (equivalent of from_mobilenet)
    @classmethod
    def from_movenet(cls, movenet_results):
        output = movenet_results

        # unwrap nested containers (same pattern you learned)
        while isinstance(output, (list, tuple)):
            output = output[0]

        output = np.asarray(output)

        # MoveNet output shapes:
        # (1,1,17,3) OR (1,17,3)
        if output.ndim == 4:
            kps = output[0][0]
        elif output.ndim == 3:
            kps = output[0]
        else:
            raise ValueError(f"Invalid MoveNet output shape: {output.shape}")

        # kps → (17,3) → y,x,score
        y = kps[:, 0]
        x = kps[:, 1]
        score = kps[:, 2]

        keypoints = np.stack([x, y], axis=1)

        return cls(
            keypoints=keypoints,
            confidence=score
        )             
    
    