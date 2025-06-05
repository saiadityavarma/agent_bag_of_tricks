import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from unified_platform.unified_sdk import UnifiedAgentPlatform
from unified_platform.stubs import AzureStubClient, BedrockStubClient, GCPStubClient


def test_basic_workflow():
    azure = AzureStubClient()
    bedrock = BedrockStubClient()
    gcp = GCPStubClient()

    platform = UnifiedAgentPlatform(azure_client=azure, bedrock_client=bedrock, gcp_client=gcp)

    azure_agent = platform.create_agent("azure-agent", provider="azure")
    bedrock_agent = platform.create_agent("bedrock-agent", provider="bedrock")
    gcp_agent = platform.create_agent("gcp-agent", provider="gcp")

    thread = platform.create_thread(azure_agent, provider="azure")
    assert thread in azure.threads

    platform.add_function(azure_agent, {"name": "echo"}, provider="azure")
    assert {"name": "echo"} in azure.agents[azure_agent]["functions"]

    platform.enable_telemetry(azure_agent, {"level": "basic"}, provider="azure")
    assert azure.agents[azure_agent]["telemetry"] == {"level": "basic"}

    platform.connect_agent(azure_agent, bedrock_agent, provider="azure")
    assert bedrock_agent in azure.agents[azure_agent].get("connections", [])

