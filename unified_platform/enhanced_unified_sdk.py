from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

class CloudProvider(Enum):
    AWS_BEDROCK = "aws_bedrock"
    GCP_VERTEX = "gcp_vertex"
    AZURE_AI = "azure_ai"

class AgentCapability(Enum):
    TOOL_CALLING = "tool_calling"
    RAG = "rag"
    CODE_INTERPRETER = "code_interpreter"

@dataclass
class AgentConfig:
    name: str
    description: str = ""
    model: str = "default"
    system_prompt: str = ""
    guardrails: Optional[Dict[str, Any]] = None
    tools: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_bases: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    max_tokens: int = 1024
    temperature: float = 0.7

@dataclass
class AgentResponse:
    agent_id: str
    provider: CloudProvider
    response_text: str
    function_calls: List[Dict[str, Any]] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseAgentClient:
    def __init__(self, provider: CloudProvider):
        self.provider = provider
        self.agents: Dict[str, Any] = {}
        self.threads: Dict[str, Any] = {}

    async def create_agent(self, config: AgentConfig) -> str:
        raise NotImplementedError

    async def invoke_agent(self, agent_id: str, message: str, thread_id: Optional[str] = None) -> AgentResponse:
        raise NotImplementedError

    async def create_thread(self, agent_id: str) -> str:
        raise NotImplementedError

    async def add_tool(self, agent_id: str, tool_spec: Dict[str, Any]) -> bool:
        raise NotImplementedError

    async def enable_rag(self, agent_id: str, knowledge_base_id: str) -> bool:
        raise NotImplementedError
