class UnifiedAgentPlatform:
    """Unified interface for Azure, Bedrock, and GCP agent SDKs."""

    def __init__(self, azure_client=None, bedrock_client=None, gcp_client=None):
        self.azure_client = azure_client
        self.bedrock_client = bedrock_client
        self.gcp_client = gcp_client

    def _get_client(self, provider):
        if provider == 'azure':
            return self.azure_client
        if provider == 'bedrock':
            return self.bedrock_client
        if provider == 'gcp':
            return self.gcp_client
        raise ValueError(f"Unsupported provider: {provider}")

    def create_agent(self, name, provider='azure', **kwargs):
        client = self._get_client(provider)
        if not client:
            raise RuntimeError(f"{provider} client not configured")
        if hasattr(client, 'create_agent'):
            return client.create_agent(name, **kwargs)
        raise NotImplementedError(f"create_agent not implemented for {provider}")

    def create_thread(self, agent_id, provider='azure', **kwargs):
        client = self._get_client(provider)
        if not client:
            raise RuntimeError(f"{provider} client not configured")
        if hasattr(client, 'create_thread'):
            return client.create_thread(agent_id, **kwargs)
        raise NotImplementedError(f"create_thread not implemented for {provider}")

    def add_function(self, agent_id, func_spec, provider='azure', **kwargs):
        client = self._get_client(provider)
        if not client:
            raise RuntimeError(f"{provider} client not configured")
        if hasattr(client, 'add_function'):
            return client.add_function(agent_id, func_spec, **kwargs)
        raise NotImplementedError(f"add_function not implemented for {provider}")

    def enable_telemetry(self, agent_id, telemetry_config, provider='azure', **kwargs):
        client = self._get_client(provider)
        if not client:
            raise RuntimeError(f"{provider} client not configured")
        if hasattr(client, 'enable_telemetry'):
            return client.enable_telemetry(agent_id, telemetry_config, **kwargs)
        raise NotImplementedError(f"enable_telemetry not implemented for {provider}")

    def connect_agent(self, agent_id, other_agent_id, provider='azure', **kwargs):
        client = self._get_client(provider)
        if not client:
            raise RuntimeError(f"{provider} client not configured")
        if hasattr(client, 'connect_agent'):
            return client.connect_agent(agent_id, other_agent_id, **kwargs)
        raise NotImplementedError(f"connect_agent not implemented for {provider}")

