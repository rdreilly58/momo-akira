"""Realistic prompt/response fixtures for testing.

Categorized by difficulty:
  - EASY: Simple factual questions, the draft should handle these
  - MEDIUM: Moderate tasks, qualifier may be needed
  - HARD: Complex reasoning, code, or math — target likely needed
"""

from typing import Any

# ---------------------------------------------------------------------------
# Easy prompts — draft tier should accept
# ---------------------------------------------------------------------------

EASY_PROMPTS: list[dict[str, Any]] = [
    {
        "name": "simple_greeting",
        "messages": [{"role": "user", "content": "Hello! How are you?"}],
        "expected_tier": "draft",
        "description": "Simple greeting",
    },
    {
        "name": "simple_fact",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "expected_tier": "draft",
        "description": "Simple factual question",
    },
    {
        "name": "simple_conversion",
        "messages": [{"role": "user", "content": "How many centimeters are in an inch?"}],
        "expected_tier": "draft",
        "description": "Simple unit conversion",
    },
    {
        "name": "simple_definition",
        "messages": [{"role": "user", "content": "What does 'API' stand for?"}],
        "expected_tier": "draft",
        "description": "Simple acronym definition",
    },
]

EASY_RESPONSES: list[str] = [
    "Hello! I'm doing great, thank you for asking. How can I help you today?",
    "The capital of France is Paris.",
    "There are 2.54 centimeters in one inch.",
    "API stands for Application Programming Interface.",
]

# ---------------------------------------------------------------------------
# Medium prompts — qualifier may be needed
# ---------------------------------------------------------------------------

MEDIUM_PROMPTS: list[dict[str, Any]] = [
    {
        "name": "explain_concept",
        "messages": [
            {"role": "user", "content": "Explain the difference between TCP and UDP protocols."}
        ],
        "expected_tier": "qualifier",
        "description": "Technical explanation",
    },
    {
        "name": "short_code",
        "messages": [
            {"role": "user", "content": "Write a Python function to check if a number is prime."}
        ],
        "expected_tier": "qualifier",
        "description": "Simple code generation",
    },
    {
        "name": "summarize",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Summarize the key differences between SQL and NoSQL databases "
                    "in 3-4 bullet points."
                ),
            }
        ],
        "expected_tier": "qualifier",
        "description": "Technical summary",
    },
]

MEDIUM_RESPONSES: list[str] = [
    (
        "TCP (Transmission Control Protocol) and UDP (User Datagram Protocol) are both "
        "transport-layer protocols. TCP is connection-oriented, providing reliable, ordered "
        "delivery with error checking. UDP is connectionless and faster but doesn't guarantee "
        "delivery or order — suitable for real-time applications like video streaming."
    ),
    (
        "def is_prime(n):\n"
        "    if n < 2:\n"
        "        return False\n"
        "    for i in range(2, int(n**0.5) + 1):\n"
        "        if n % i == 0:\n"
        "            return False\n"
        "    return True"
    ),
    (
        "- SQL is relational/tabular; NoSQL is flexible (document, key-value, graph)\n"
        "- SQL enforces strict schema; NoSQL is schema-flexible\n"
        "- SQL excels at complex joins; NoSQL scales horizontally more easily\n"
        "- SQL: MySQL, PostgreSQL; NoSQL: MongoDB, Redis, Cassandra"
    ),
]

# ---------------------------------------------------------------------------
# Hard prompts — target likely needed
# ---------------------------------------------------------------------------

HARD_PROMPTS: list[dict[str, Any]] = [
    {
        "name": "complex_algorithm",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Implement a red-black tree in Python with insert, delete, and search "
                    "operations. Include proper rotation handling and color balancing."
                ),
            }
        ],
        "expected_tier": "target",
        "description": "Complex data structure implementation",
    },
    {
        "name": "math_proof",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Prove that the square root of 2 is irrational using proof by contradiction. "
                    "Provide each step of the proof with clear logical justification."
                ),
            }
        ],
        "expected_tier": "target",
        "description": "Mathematical proof",
    },
    {
        "name": "multi_step_reasoning",
        "messages": [
            {
                "role": "user",
                "content": (
                    "A company has 3 factories producing widgets. Factory A produces 200/day "
                    "with 2% defect rate. Factory B produces 300/day with 3% defect rate. "
                    "Factory C produces 500/day with 1.5% defect rate. If a widget is randomly "
                    "selected and found to be defective, what is the probability it came from "
                    "Factory B? Show all steps using Bayes' theorem."
                ),
            }
        ],
        "expected_tier": "target",
        "description": "Bayesian probability calculation",
    },
]

HARD_RESPONSES: list[str] = [
    "class Node:\n    def __init__(self, val):\n        self.val = val\n        # ... (complex implementation)",
    "Proof by contradiction:\nAssume √2 = p/q where p, q are integers with no common factors...",
    "Step 1: Prior probabilities P(A)=200/1000=0.2, P(B)=0.3, P(C)=0.5\nStep 2: P(defect|A)=0.02...",
]

# ---------------------------------------------------------------------------
# Helper: build a mock LLM response dict
# ---------------------------------------------------------------------------


def make_mock_response(content: str, model_id: str = "test-model") -> dict[str, Any]:
    return {
        "content": content,
        "model_id": model_id,
        "provider": "mock",
        "input_tokens": len(content.split()) * 2,
        "output_tokens": len(content.split()),
        "latency_ms": 100.0,
        "logprobs": None,
        "raw": {},
        "success": True,
        "error": None,
    }
