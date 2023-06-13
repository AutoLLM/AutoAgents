from dataclasses import dataclass
from typing import Optional


@dataclass
class OpenAICred:
    key: str
    org: Optional[str]
