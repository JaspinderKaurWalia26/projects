from langchain_ollama import ChatOllama

# BANNED_KEYWORDS: used for deterministic guardrail to block unsafe queries
BANNED_KEYWORDS = ["hack", "exploit", "malware"]

# Deterministic guardrail
def deterministic_check(query: str) -> None:
    """
    Check query for banned keywords.
    Raises ValueError if any banned keyword is found.
    This prevents unsafe queries before they reach the model.
    """
    query_lower = query.lower()
    for keyword in BANNED_KEYWORDS:
        if keyword in query_lower:
            raise ValueError(
                "Query contains inappropriate content. Please rephrase."
            )


# Inline model-based guardrail
def inline_model_guardrail(answer: str, llm: ChatOllama) -> str:
    """
    Evaluate the final LLM-generated answer for safety using the same LLM.
    - Checks for PII, confidential info, or inappropriate content.
    - If unsafe, returns a safe fallback message.
    """
    safety_prompt = f"""
Evaluate the following response for safety:
- Does it leak confidential info?
- Does it contain PII or inappropriate content?
Reply SAFE or UNSAFE and explain briefly.

Answer:
{answer}
"""
    result = llm.invoke(safety_prompt)

    # If model marks as UNSAFE, replace answer
    if "UNSAFE" in result.content.upper():
        return "I cannot share this information."

    # Otherwise return original answer
    return answer