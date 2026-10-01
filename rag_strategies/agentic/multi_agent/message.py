from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentMessage:
    sender: str
    recipient: str
    message_type: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
