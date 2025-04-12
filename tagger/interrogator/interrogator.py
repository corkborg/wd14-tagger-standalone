import re
import sys

import pandas as pd

from typing import Iterable, Tuple, List, Dict
from PIL import Image

from onnxruntime import InferenceSession, get_available_providers

tag_escape_pattern = re.compile(r'([\\()])')

class AbsInterrogator:
    model: InferenceSession | None
    tags: pd.DataFrame | None
    @staticmethod
    def postprocess_tags(
        tags: Dict[str, float],
        threshold=0.35,
        additional_tags: List[str] = [],
        exclude_tags: Iterable[str] = [],
        sort_by_alphabetical_order=False,
        add_confident_as_weight=False,
        replace_underscore=False,
        replace_underscore_excludes: List[str] = [],
        escape_tag=False
    ) -> Dict[str, float]:
        for t in additional_tags:
            tags[t] = 1.0

        # those lines are totally not "pythonic" but looks better to me
        tags = {
            t: c

            # sort by tag name or confident
            for t, c in sorted(
                tags.items(),
                key=lambda i: i[0 if sort_by_alphabetical_order else 1],
                reverse=not sort_by_alphabetical_order
            )

            # filter tags
            if (
                c >= threshold
                and t not in exclude_tags
            )
        }

        new_tags = []
        for tag in list(tags):
            new_tag = tag

            if replace_underscore and tag not in replace_underscore_excludes:
                new_tag = new_tag.replace('_', ' ')

            if escape_tag:
                new_tag = tag_escape_pattern.sub(r'\\\1', new_tag)

            if add_confident_as_weight:
                new_tag = f'({new_tag}:{tags[tag]})'

            new_tags.append((new_tag, tags[tag]))
        tags = dict(new_tags)

        return tags

    def __init__(self, name: str) -> None:
        self.name = name
        # Initialize with optimal provider selection
        self.providers = self.get_optimal_provider()

    def load(self):
        raise NotImplementedError()

    def unload(self) -> bool:
        unloaded = False

        if hasattr(self, 'model') and self.model is not None:
            del self.model
            unloaded = True
            print(f'Unloaded {self.name}', file=sys.stderr)

        if hasattr(self, 'tags'):
            del self.tags

        return unloaded

    def use_cpu(self) -> None:
        """Force CPU-only execution."""
        self.providers = ['CPUExecutionProvider']
        print(f'Forcing CPU execution for {self.name}', file=sys.stderr)

    def get_available_providers(self) -> List[str]:
        """Get list of available execution providers."""
        return get_available_providers()

    def set_providers(self, providers: List[str]) -> None:
        """Set specific execution providers."""
        self.providers = providers
        #print(f'Set custom providers for {self.name}: {providers}', file=sys.stderr)

    def get_optimal_provider(self) -> List[str]:
        """Get the optimal provider based on system capabilities.
        
        Returns a list of providers in order of preference:
        - CoreMLExecutionProvider (if on Apple Silicon)
        - CUDAExecutionProvider (if NVIDIA GPU available)
        - CPUExecutionProvider (always available as fallback)
        """
        available = self.get_available_providers()
        
        # Start with most optimal providers first
        optimal_order = [
            'CoreMLExecutionProvider',  # Best for Apple Silicon
            'CUDAExecutionProvider',    # Best for NVIDIA GPUs
            'CPUExecutionProvider'      # Fallback
        ]
        
        # Return list of available providers in optimal order
        selected = [p for p in optimal_order if p in available]
        #print(f'Selected optimal providers for {self.name}: {selected}', file=sys.stderr)
        return selected

    def interrogate(
        self,
        image: Image.Image
    ) -> Tuple[
        Dict[str, float],  # rating confidents
        Dict[str, float]  # tag confidents
    ]:
        raise NotImplementedError()
