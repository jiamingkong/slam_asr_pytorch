from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from torch import Tensor


@dataclass
class DataCollatorForSlamASR(object):
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, Tensor]:
        # Extract elements
        # x = ds["speech"][0:3]
        # y = ds["text"][0:3]

        x = [i["speech"] for i in instances]
        y = [i["text"].lower() for i in instances]
        return {"audios": x, "transcriptions": y}
