class BaseStubClient:
    def __init__(self):
        self.agents = {}
        self.threads = {}

    def create_agent(self, name, **kwargs):
        agent_id = f"agent-{len(self.agents)+1}"
        self.agents[agent_id] = {'name': name, 'functions': [], 'telemetry': None}
        return agent_id

    def create_thread(self, agent_id, **kwargs):
        thread_id = f"thread-{len(self.threads)+1}"
        self.threads[thread_id] = {'agent_id': agent_id, 'messages': []}
        return thread_id

    def add_function(self, agent_id, func_spec, **kwargs):
        self.agents[agent_id]['functions'].append(func_spec)
        return True

    def enable_telemetry(self, agent_id, telemetry_config, **kwargs):
        self.agents[agent_id]['telemetry'] = telemetry_config
        return True

    def connect_agent(self, agent_id, other_agent_id, **kwargs):
        connection = self.agents[agent_id].setdefault('connections', [])
        connection.append(other_agent_id)
        return True

class AzureStubClient(BaseStubClient):
    pass

class BedrockStubClient(BaseStubClient):
    pass

class GCPStubClient(BaseStubClient):
    pass

