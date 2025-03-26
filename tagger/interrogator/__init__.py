import re
import sys
from typing import Iterable

import onnxruntime
import pandas as pd
from PIL import Image

tag_escape_pattern = re.compile(r'([\\()])')

class AbsInterrogator:
    model: onnxruntime.InferenceSession | None
    tags: pd.DataFrame | None
    @staticmethod
    def postprocess_tags(
        tags: dict[str, float],
        threshold=0.35,
        additional_tags: list[str] = [],
        exclude_tags: Iterable[str] = [],
        sort_by_alphabetical_order=False,
        add_confident_as_weight=False,
        replace_underscore=False,
        replace_underscore_excludes: list[str] = [],
        escape_tag=False
    ) -> dict[str, float]:
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
        self.providers = onnxruntime.get_available_providers()

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
        self.providers = ['CPUExecutionProvider']

    def override_execution_provider(
        self, execution_providers: list[str]
    ):
        self.providers = execution_providers

    def interrogate(
        self,
        image: Image.Image
    ) -> tuple[
        dict[str, float],  # rating confidents
        dict[str, float]  # tag confidents
    ]:
        raise NotImplementedError()
