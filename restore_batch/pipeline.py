from __future__ import annotations

from typing import Protocol

from .models import ImageContext


class PipelineStage(Protocol):
    name: str

    def process(self, context: ImageContext) -> ImageContext:
        ...


class ProcessingPipeline:
    def __init__(self, stages: list[PipelineStage]) -> None:
        self._stages = stages

    def run(self, context: ImageContext) -> ImageContext:
        current = context
        for stage in self._stages:
            current = stage.process(current)
        return current
