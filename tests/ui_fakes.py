"""A canned pipeline for UI tests. Tests put a real result in RESULT first."""
from tests.conftest import make_config

RESULT: dict = {}


class FakePipeline:
    def describe(self):
        return make_config().describe()

    def ask_stream(self, question):
        yield {"event": "node", "node": "plan_tools",
               "update": {"tool_plan": [{"name": "search_trials", "args": {"condition": "diabetes"}}]}}
        yield {"event": "node", "node": "execute_tools", "update": {"tool_runs": [{"n_trials": 2}]}}
        yield {"event": "result", "result": RESULT}


class FakePipelineWithOllama(FakePipeline):
    def describe(self):
        info = super().describe()
        for entry in (info["planner"], *info["synthesizers"], info["evidence"], info["critic"]):
            entry["base_url"] = "http://ollama.test:11434"
        return info
