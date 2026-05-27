"""
Abstract base class for all VLM wrappers.
Every model wrapper must implement predict().
"""
from abc import ABC, abstractmethod
from typing import Any


class BaseVLM(ABC):

    @abstractmethod
    def load(self) -> None:
        """Load model and processor into memory."""
        ...

    @abstractmethod
    def predict(self, request: Any) -> str:
        """
        Given a FocusDataset request, return a text answer string.
        Must respect the track's time budget.
        """
        ...

    def predict_batch(self, requests: list) -> list[str]:
        """Default: loop predict(). Override for true batching."""
        return [self.predict(r) for r in requests]
