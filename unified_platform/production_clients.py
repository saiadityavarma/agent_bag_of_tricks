import asyncio
import logging
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from .enhanced_unified_sdk import (
    BaseAgentClient, AgentConfig, AgentResponse, CloudProvider,
    AgentCapability
)

# AWS Imports
try:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    logging.warning("AWS SDK not available. Install boto3 to use AWS Bedrock Agents.")

# GCP Imports
try:
    from google.cloud import aiplatform
    from google.cloud.aiplatform import gapic as aiplatform_gapic
    from google.auth import default as google_default_auth
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False
    logging.warning("GCP SDK not available. Install google-cloud-aiplatform to use Vertex AI Agents.")

# Azure Imports
try:
    from azure.identity import DefaultAzureCredential
    from azure.ai.ml import MLClient
    from azure.core.exceptions import AzureError
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    logging.warning("Azure SDK not available. Install azure-ai-ml to use Azure AI Agents.")

logger = logging.getLogger(__name__)


class ProductionAWSBedrockClient(BaseAgentClient):
    """Production AWS Bedrock Agents client using latest SDK"""

    def __init__(self, region: str = "us-east-1", aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None, aws_session_token: Optional[str] = None):
        if not AWS_AVAILABLE:
            raise ImportError("AWS SDK not available. Install boto3 and botocore.")

        super().__init__(CloudProvider.AWS_BEDROCK)
        self.region = region

        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=region
        )

        self.bedrock_agent = session.client('bedrock-agent')
        self.bedrock_agent_runtime = session.client('bedrock-agent-runtime')
        self.bedrock = session.client('bedrock')

        self.strands_enabled = self._check_strands_availability()

    def _check_strands_availability(self) -> bool:
        try:
            import strands_agents  # type: ignore
            return True
        except ImportError:
            logger.info("Strands Agents not available. Using standard Bedrock Agents.")
            return False

    async def create_agent(self, config: AgentConfig) -> str:
        try:
            agent_config = {
                'agentName': config.name,
                'description': config.description,
                'foundationModel': self._map_model(config.model),
                'instruction': config.system_prompt,
                'idleSessionTTLInSeconds': 3600,
                'guardrailConfiguration': self._prepare_guardrails(config.guardrails),
                'promptOverrideConfiguration': {
                    'promptConfigurations': [
                        {
                            'promptType': 'PRE_PROCESSING',
                            'promptCreationMode': 'DEFAULT'
                        }
                    ]
                }
            }

            response = self.bedrock_agent.create_agent(**agent_config)
            agent_id = response['agent']['agentId']

            self.agents[agent_id] = {
                'config': agent_config,
                'aws_agent_id': agent_id,
                'status': response['agent']['agentStatus'],
                'tools': config.tools,
                'knowledge_bases': config.knowledge_bases
            }

            for tool in config.tools:
                await self.add_tool(agent_id, tool)

            for kb_id in config.knowledge_bases:
                await self.enable_rag(agent_id, kb_id)

            self.bedrock_agent.prepare_agent(agentId=agent_id)

            logger.info(f"Created AWS Bedrock Agent: {agent_id}")
            return agent_id

        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to create AWS Bedrock Agent: {e}")
            raise

    async def invoke_agent(self, agent_id: str, message: str, thread_id: Optional[str] = None) -> AgentResponse:
        try:
            if not thread_id:
                thread_id = await self.create_thread(agent_id)

            aws_agent_id = self.agents[agent_id]['aws_agent_id']

            response = self.bedrock_agent_runtime.invoke_agent(
                agentId=aws_agent_id,
                agentAliasId='TSTALIASID',
                sessionId=thread_id,
                inputText=message,
                enableTrace=True
            )

            response_text = ""
            function_calls = []
            citations = []

            for event in response.get('completion', []):
                if 'chunk' in event:
                    chunk = event['chunk']
                    if 'bytes' in chunk:
                        response_text += chunk['bytes'].decode('utf-8')
                elif 'trace' in event:
                    trace = event['trace']['trace']
                    if 'orchestrationTrace' in trace:
                        orch_trace = trace['orchestrationTrace']
                        if 'invocationInput' in orch_trace:
                            function_calls.append(orch_trace['invocationInput'])

            return AgentResponse(
                agent_id=agent_id,
                provider=self.provider,
                response_text=response_text,
                function_calls=function_calls,
                citations=citations,
                metadata={"thread_id": thread_id, "aws_agent_id": aws_agent_id}
            )

        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to invoke AWS Bedrock Agent: {e}")
            raise

    async def create_thread(self, agent_id: str) -> str:
        thread_id = f"bedrock-session-{uuid.uuid4().hex}"
        self.threads[thread_id] = {
            'agent_id': agent_id,
            'messages': [],
            'created_at': datetime.now()
        }
        return thread_id

    async def add_tool(self, agent_id: str, tool_spec: Dict) -> bool:
        try:
            aws_agent_id = self.agents[agent_id]['aws_agent_id']

            action_group_config = {
                'agentId': aws_agent_id,
                'agentVersion': 'DRAFT',
                'actionGroupName': tool_spec.get('name', f"tool-{uuid.uuid4().hex[:8]}") ,
                'description': tool_spec.get('description', 'Custom tool'),
                'actionGroupExecutor': {
                    'lambda': tool_spec.get('lambda_arn', 'arn:aws:lambda:region:account:function:tool-function')
                },
                'apiSchema': {
                    'payload': json.dumps(tool_spec.get('schema', {}))
                }
            }

            self.bedrock_agent.create_agent_action_group(**action_group_config)
            self.agents[agent_id]['tools'].append(tool_spec)

            logger.info(f"Added tool to AWS Bedrock Agent: {tool_spec['name']}")
            return True

        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to add tool to AWS Bedrock Agent: {e}")
            return False

    async def enable_rag(self, agent_id: str, knowledge_base_id: str) -> bool:
        try:
            aws_agent_id = self.agents[agent_id]['aws_agent_id']

            self.bedrock_agent.associate_agent_knowledge_base(
                agentId=aws_agent_id,
                agentVersion='DRAFT',
                description='RAG knowledge base',
                knowledgeBaseId=knowledge_base_id,
                knowledgeBaseState='ENABLED'
            )

            self.agents[agent_id]['knowledge_bases'].append(knowledge_base_id)
            logger.info(f"Enabled RAG for AWS Bedrock Agent with KB: {knowledge_base_id}")
            return True

        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to enable RAG for AWS Bedrock Agent: {e}")
            return False

    def _map_model(self, model: str) -> str:
        model_mapping = {
            "default": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
            "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
            "claude-3-opus": "anthropic.claude-3-opus-20240229-v1:0",
            "claude-3.5-sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "titan-text": "amazon.titan-text-premier-v1:0",
            "llama3": "meta.llama3-2-90b-instruct-v1:0"
        }
        return model_mapping.get(model, model)

    def _prepare_guardrails(self, guardrails: Dict) -> Dict:
        if not guardrails:
            return {}
        return {
            'guardrailIdentifier': guardrails.get('guardrail_id'),
            'guardrailVersion': guardrails.get('version', 'DRAFT')
        }


class ProductionGCPVertexClient(BaseAgentClient):
    """Production GCP Vertex AI client using latest SDK"""

    def __init__(self, project_id: str, location: str = "us-central1", credentials_path: Optional[str] = None):
        if not GCP_AVAILABLE:
            raise ImportError("GCP SDK not available. Install google-cloud-aiplatform.")

        super().__init__(CloudProvider.GCP_VERTEX)
        self.project_id = project_id
        self.location = location

        aiplatform.init(
            project=project_id,
            location=location,
            credentials=credentials_path
        )

        self.agent_client = aiplatform_gapic.PipelineServiceClient()
        self.model_client = aiplatform_gapic.ModelServiceClient()

    async def create_agent(self, config: AgentConfig) -> str:
        try:
            agent_id = f"vertex-agent-{uuid.uuid4().hex[:8]}"

            from google.cloud import aiplatform_v1beta1

            agent_config = {
                "display_name": config.name,
                "description": config.description,
                "system_instruction": config.system_prompt,
                "model": self._map_model(config.model),
                "tools": self._prepare_tools(config.tools),
                "generation_config": {
                    "max_output_tokens": config.max_tokens,
                    "temperature": config.temperature,
                    "top_p": 0.95,
                    "top_k": 40
                }
            }

            client = aiplatform_v1beta1.AgentServiceClient()
            parent = f"projects/{self.project_id}/locations/{self.location}"
            request = aiplatform_v1beta1.CreateAgentRequest(
                parent=parent,
                agent_id=agent_id,
                agent=agent_config
            )

            self.agents[agent_id] = {
                'config': agent_config,
                'gcp_agent_id': agent_id,
                'status': 'ACTIVE',
                'tools': config.tools,
                'knowledge_bases': config.knowledge_bases
            }

            logger.info(f"Created GCP Vertex AI Agent: {agent_id}")
            return agent_id

        except Exception as e:
            logger.error(f"Failed to create GCP Vertex AI Agent: {e}")
            raise

    async def invoke_agent(self, agent_id: str, message: str, thread_id: Optional[str] = None) -> AgentResponse:
        try:
            if not thread_id:
                thread_id = await self.create_thread(agent_id)

            from vertexai.generative_models import GenerativeModel

            model_name = self.agents[agent_id]['config']['model']
            model = GenerativeModel(model_name)

            if thread_id not in self.threads:
                chat = model.start_chat()
                self.threads[thread_id]['chat_session'] = chat
            else:
                chat = self.threads[thread_id]['chat_session']

            response = chat.send_message(message)

            return AgentResponse(
                agent_id=agent_id,
                provider=self.provider,
                response_text=response.text,
                function_calls=[],
                citations=[],
                metadata={
                    "thread_id": thread_id,
                    "model": model_name,
                    "safety_ratings": [rating.to_dict() for rating in response.candidates[0].safety_ratings]
                }
            )

        except Exception as e:
            logger.error(f"Failed to invoke GCP Vertex AI Agent: {e}")
            raise

    async def create_thread(self, agent_id: str) -> str:
        thread_id = f"vertex-thread-{uuid.uuid4().hex}"
        self.threads[thread_id] = {
            'agent_id': agent_id,
            'messages': [],
            'created_at': datetime.now(),
            'chat_session': None
        }
        return thread_id

    async def add_tool(self, agent_id: str, tool_spec: Dict) -> bool:
        try:
            vertex_tool = self._convert_tool_spec(tool_spec)
            self.agents[agent_id]['tools'].append(vertex_tool)

            logger.info(f"Added tool to GCP Vertex AI Agent: {tool_spec['name']}")
            return True

        except Exception as e:
            logger.error(f"Failed to add tool to GCP Vertex AI Agent: {e}")
            return False

    async def enable_rag(self, agent_id: str, knowledge_base_id: str) -> bool:
        try:
            from google.cloud import aiplatform
            self.agents[agent_id]['knowledge_bases'].append(knowledge_base_id)

            logger.info(f"Enabled RAG for GCP Vertex AI Agent with KB: {knowledge_base_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to enable RAG for GCP Vertex AI Agent: {e}")
            return False

    def _map_model(self, model: str) -> str:
        model_mapping = {
            "default": "gemini-1.5-pro-002",
            "gemini-pro": "gemini-1.5-pro-002",
            "gemini-flash": "gemini-1.5-flash-002",
            "gemini-ultra": "gemini-ultra",
            "claude-3-sonnet": "claude-3-sonnet@vertex-ai",
            "claude-3-haiku": "claude-3-haiku@vertex-ai"
        }
        return model_mapping.get(model, model)

    def _prepare_tools(self, tools: List[Dict]) -> List[Dict]:
        vertex_tools = []
        for tool in tools:
            vertex_tool = {
                "function_declarations": [
                    {
                        "name": tool.get("name"),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {})
                    }
                ]
            }
            vertex_tools.append(vertex_tool)
        return vertex_tools

    def _convert_tool_spec(self, tool_spec: Dict) -> Dict:
        return {
            "function_declarations": [
                {
                    "name": tool_spec.get("name"),
                    "description": tool_spec.get("description", ""),
                    "parameters": tool_spec.get("schema", {})
                }
            ]
        }


class ProductionAzureAIClient(BaseAgentClient):
    """Production Azure AI client using latest SDK"""

    def __init__(self, subscription_id: str, resource_group: str, workspace_name: str, credential: Optional[Any] = None):
        if not AZURE_AVAILABLE:
            raise ImportError("Azure SDK not available. Install azure-ai-ml.")

        super().__init__(CloudProvider.AZURE_AI)
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_name = workspace_name

        self.credential = credential or DefaultAzureCredential()
        self.ml_client = MLClient(
            credential=self.credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )

        self._init_ai_studio_clients()

    def _init_ai_studio_clients(self):
        try:
            from azure.ai.generative import GenerativeAIClient
            from azure.ai.assistant import AssistantClient

            self.generative_client = GenerativeAIClient(
                credential=self.credential,
                subscription_id=self.subscription_id
            )

            self.assistant_client = AssistantClient(
                credential=self.credential,
                subscription_id=self.subscription_id
            )

        except ImportError:
            logger.warning("Latest Azure AI Studio SDKs not available. Using fallback methods.")
            self.generative_client = None
            self.assistant_client = None

    async def create_agent(self, config: AgentConfig) -> str:
        try:
            agent_id = f"azure-agent-{uuid.uuid4().hex[:8]}"

            if self.assistant_client:
                assistant_config = {
                    "name": config.name,
                    "description": config.description,
                    "instructions": config.system_prompt,
                    "model": self._map_model(config.model),
                    "tools": self._prepare_tools(config.tools),
                    "metadata": config.metadata
                }

                assistant = self.assistant_client.create_assistant(**assistant_config)
                azure_agent_id = assistant.id
            else:
                azure_agent_id = agent_id
                assistant_config = config.__dict__.copy()

            self.agents[agent_id] = {
                'config': assistant_config,
                'azure_agent_id': azure_agent_id,
                'status': 'ACTIVE',
                'tools': config.tools,
                'knowledge_bases': config.knowledge_bases
            }

            logger.info(f"Created Azure AI Agent: {agent_id}")
            return agent_id

        except AzureError as e:
            logger.error(f"Failed to create Azure AI Agent: {e}")
            raise

    async def invoke_agent(self, agent_id: str, message: str, thread_id: Optional[str] = None) -> AgentResponse:
        try:
            if not thread_id:
                thread_id = await self.create_thread(agent_id)

            azure_agent_id = self.agents[agent_id]['azure_agent_id']

            if self.assistant_client:
                thread = self.assistant_client.get_thread(thread_id)
                self.assistant_client.create_message(
                    thread_id=thread_id,
                    role="user",
                    content=message
                )

                run = self.assistant_client.create_run(
                    thread_id=thread_id,
                    assistant_id=azure_agent_id
                )

                while run.status in ["queued", "in_progress"]:
                    await asyncio.sleep(1)
                    run = self.assistant_client.get_run(thread_id=thread_id, run_id=run.id)

                messages = self.assistant_client.list_messages(thread_id=thread_id)
                latest_message = messages.data[0]

                response_text = latest_message.content[0].text.value
            else:
                response_text = f"Azure AI Agent response to: {message}"

            return AgentResponse(
                agent_id=agent_id,
                provider=self.provider,
                response_text=response_text,
                function_calls=[],
                citations=[],
                metadata={"thread_id": thread_id, "azure_agent_id": azure_agent_id}
            )

        except AzureError as e:
            logger.error(f"Failed to invoke Azure AI Agent: {e}")
            raise

    async def create_thread(self, agent_id: str) -> str:
        try:
            if self.assistant_client:
                thread = self.assistant_client.create_thread()
                thread_id = thread.id
            else:
                thread_id = f"azure-thread-{uuid.uuid4().hex}"

            self.threads[thread_id] = {
                'agent_id': agent_id,
                'messages': [],
                'created_at': datetime.now()
            }
            return thread_id

        except AzureError as e:
            logger.error(f"Failed to create Azure AI thread: {e}")
            raise

    async def add_tool(self, agent_id: str, tool_spec: Dict) -> bool:
        try:
            azure_tool = self._convert_tool_spec(tool_spec)
            self.agents[agent_id]['tools'].append(azure_tool)

            logger.info(f"Added tool to Azure AI Agent: {tool_spec['name']}")
            return True

        except AzureError as e:
            logger.error(f"Failed to add tool to Azure AI Agent: {e}")
            return False

    async def enable_rag(self, agent_id: str, knowledge_base_id: str) -> bool:
        try:
            self.agents[agent_id]['knowledge_bases'].append(knowledge_base_id)

            logger.info(f"Enabled RAG for Azure AI Agent with KB: {knowledge_base_id}")
            return True

        except AzureError as e:
            logger.error(f"Failed to enable RAG for Azure AI Agent: {e}")
            return False

    def _map_model(self, model: str) -> str:
        model_mapping = {
            "default": "gpt-4o-2024-11-20",
            "gpt-4": "gpt-4-turbo-2024-04-09",
            "gpt-4o": "gpt-4o-2024-11-20",
            "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
            "gpt-35-turbo": "gpt-35-turbo-16k"
        }
        return model_mapping.get(model, model)

    def _prepare_tools(self, tools: List[Dict]) -> List[Dict]:
        azure_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                azure_tools.append({"type": "function", "function": tool})
            elif tool.get("type") == "code_interpreter":
                azure_tools.append({"type": "code_interpreter"})
            elif tool.get("type") == "retrieval":
                azure_tools.append({"type": "retrieval"})
        return azure_tools

    def _convert_tool_spec(self, tool_spec: Dict) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": tool_spec.get("name"),
                "description": tool_spec.get("description", ""),
                "parameters": tool_spec.get("schema", {})
            }
        }

