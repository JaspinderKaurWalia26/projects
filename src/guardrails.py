from langchain_ollama import ChatOllama
from src.logger import setup_logger

logger = setup_logger(__name__)

# Banned keywords for deterministic guardrail
BANNED_KEYWORDS = [
    # Cybersecurity threats
    "hack", "exploit", "malware", "ransomware", "phishing",
    # Violence
    "kill", "attack", "bomb", "weapon",
    # Illegal activities
    "drug", "illegal", "steal", "fraud",
]


def deterministic_check(query: str) -> None:
    """
    Check query for banned keywords.
    Args:
        query: User input query
    Raises:
        ValueError: If query is empty or contains banned keywords
    """

    # Check if query is empty
    if not query.strip():
        logger.warning("Empty query provided")
        raise ValueError("Query cannot be empty")

    query_lower = query.lower()

    for keyword in BANNED_KEYWORDS:
        if keyword in query_lower:
            logger.warning(f"Banned keyword detected: '{keyword}' in query")
            raise ValueError(
                "Query contains inappropriate content. Please rephrase."
            )

    logger.info("Deterministic check passed")


def inline_model_guardrail(answer: str, llm: ChatOllama) -> str:
    """
    Evaluate the final LLM-generated answer for safety.
    Args:
        answer: LLM generated answer
        llm: ChatOllama instance
    Returns:
        Original answer if safe, fallback message if unsafe
    Raises:
        RuntimeError: If LLM fails to evaluate safety
    """

    # Check if answer is empty
    if not answer.strip():
        logger.warning("Empty answer provided to guardrail")
        return "No answer was generated."

    safety_prompt = f"""
Evaluate the following response for safety. Check for:
1. Confidential or sensitive information leak
2. Personal Identifiable Information (PII)
3. Inappropriate or harmful content
4. Misleading or false information

Reply with SAFE or UNSAFE followed by a brief explanation.

Answer to evaluate:
{answer}
"""

    try:
        logger.info("Running inline model guardrail check...")
        result = llm.invoke(safety_prompt)

        # If model marks as UNSAFE replace answer
        if "UNSAFE" in result.content.upper():
            logger.warning(f"Unsafe answer detected: {result.content[:100]}")
            return "I cannot share this information."

        logger.info("Inline guardrail check passed")
        return answer

    except Exception as e:
        logger.error(f"Guardrail check failed: {e}")
        raise RuntimeError(f"Guardrail check failed: {e}")