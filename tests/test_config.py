"""Tests for config loading."""

from config import load_agents_config, load_routing_rules


def test_load_agents_config():
    agents = load_agents_config()
    assert "Triage" in agents
    assert "TechnicalSupport" in agents
    assert "Billing" in agents
    assert "Escalation" in agents

    for name, config in agents.items():
        assert "system_prompt" in config
        assert "routing_rules" in config
        assert "available_tools" in config
        assert "escalation_threshold" in config


def test_load_routing_rules():
    rules = load_routing_rules()
    intents = rules["intents"]
    assert intents["technical"] == "TechnicalSupport"
    assert intents["billing"] == "Billing"
    assert intents["escalation"] == "Escalation"
    assert "unknown" in intents
