"""API Call Agent — custom reward environment for Nova RFT.

The model translates natural-language requests into structured API calls.
Uses SingleTurnEnv (no abstract methods required) to avoid the
MultiTurnEnv.env_response requirement.

Reward:
  - 0.5  correct endpoint in <action> JSON
  - 0.3  required params present
  - 0.2  valid format compliance
"""
import json
import logging
from datasets import Dataset
import verifiers as vf

logger = logging.getLogger(__name__)

# Endpoint catalog for param validation
ENDPOINT_PARAMS = {
    "/weather": ["city"],
    "/flights/book": ["origin", "destination"],
    "/restaurants/search": ["cuisine"],
    "/translate": ["text", "target"],
    "/stocks/quote": ["symbol"],
    "/email/send": ["to"],
    "/reminders/create": ["time", "message"],
    "/smart-home/lights": ["room", "action"],
    "/calculate": ["expression"],
    "/calendar/next": [],
}


def load_environment(**kwargs) -> vf.Environment:
    """Entry point called by the Nova SDK."""
    dataset = Dataset.from_list([
        {"question": "What's the weather in Seattle?", "answer": "/weather"},
        {"question": "Book a flight from SEA to JFK", "answer": "/flights/book"},
        {"question": "Find Italian restaurants nearby", "answer": "/restaurants/search"},
        {"question": "Translate 'hello' to Spanish", "answer": "/translate"},
        {"question": "What is AMZN stock price?", "answer": "/stocks/quote"},
        {"question": "Send email to the team", "answer": "/email/send"},
        {"question": "Set reminder at 3pm to call dentist", "answer": "/reminders/create"},
        {"question": "Turn off living room lights", "answer": "/smart-home/lights"},
        {"question": "Calculate 85 * 0.15", "answer": "/calculate"},
        {"question": "When is my next meeting?", "answer": "/calendar/next"},
        {"question": "What's the weather in Tokyo?", "answer": "/weather"},
        {"question": "Book flight from LAX to ORD", "answer": "/flights/book"},
        {"question": "Find Mexican restaurants nearby", "answer": "/restaurants/search"},
        {"question": "Translate 'goodbye' to French", "answer": "/translate"},
        {"question": "What is GOOGL stock price?", "answer": "/stocks/quote"},
    ])

    parser = vf.XMLParser(["think", "action"], answer_field="action")
    system_prompt = (
        "You are an API assistant. Given a user request, determine the correct "
        "API endpoint and parameters.\n"
        f"Respond in format: {parser.get_format_str()}\n"
        "Put reasoning in <think>, and a JSON object with 'endpoint' and "
        "'params' keys in <action>.\n"
        "Example: <action>{\"endpoint\": \"/weather\", \"params\": {\"city\": \"Seattle\"}}</action>"
    )

    def endpoint_reward(completion, answer, **kw) -> float:
        """0.5 if correct endpoint."""
        try:
            action_text = parser.parse(completion).get("action", "")
            if not action_text:
                return 0.0
            call = json.loads(action_text)
            return 0.5 if call.get("endpoint") == answer else 0.0
        except (json.JSONDecodeError, TypeError, AttributeError):
            return 0.0

    def params_reward(completion, answer, **kw) -> float:
        """0.3 if required params present. Only scored if endpoint is correct."""
        try:
            action_text = parser.parse(completion).get("action", "")
            if not action_text:
                return 0.0
            call = json.loads(action_text)
            if call.get("endpoint") != answer:
                return 0.0
            expected_keys = ENDPOINT_PARAMS.get(answer, [])
            if not expected_keys:
                return 0.3
            call_params = call.get("params", {})
            if not isinstance(call_params, dict):
                return 0.0
            present = sum(1 for k in expected_keys if k in call_params)
            return 0.3 * (present / len(expected_keys))
        except (json.JSONDecodeError, TypeError, AttributeError):
            return 0.0

    rubric = vf.Rubric(
        parser=parser,
        funcs=[endpoint_reward, params_reward, parser.get_format_reward_func()],
        weights=[1.0, 1.0, 0.2],
    )

    return vf.SingleTurnEnv(
        eval_dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        max_concurrent=10,
    )
