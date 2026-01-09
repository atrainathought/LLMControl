"""
Prompt Engineering Examples

This module demonstrates three key prompting techniques:
1. Zero-shot: Direct question without examples
2. Few-shot: Include examples to guide the model
3. Chain-of-thought: Ask the model to reason step-by-step

We use a sentiment classification task to compare these approaches.
"""

# =============================================================================
# TASK: Sentiment Classification
# Given a product review, classify it as POSITIVE, NEGATIVE, or NEUTRAL
# =============================================================================

# -----------------------------------------------------------------------------
# 1. ZERO-SHOT PROMPTING
# -----------------------------------------------------------------------------
# Simply ask the model to perform the task with no examples.
# Relies on the model's pre-trained knowledge.

ZERO_SHOT_PROMPT = """Classify the sentiment of the following product review as POSITIVE, NEGATIVE, or NEUTRAL.

Review: {review}

Respond with only one word: POSITIVE, NEGATIVE, or NEUTRAL."""


# -----------------------------------------------------------------------------
# 2. FEW-SHOT PROMPTING
# -----------------------------------------------------------------------------
# Provide examples to guide the model's behavior.
# The model learns the pattern from the examples.

FEW_SHOT_PROMPT = """Classify the sentiment of product reviews as POSITIVE, NEGATIVE, or NEUTRAL.

Example 1:
Review: "This product exceeded my expectations! Great quality and fast shipping."
Sentiment: POSITIVE

Example 2:
Review: "Terrible experience. The item arrived broken and customer service was unhelpful."
Sentiment: NEGATIVE

Example 3:
Review: "It works as described. Nothing special but does the job."
Sentiment: NEUTRAL

Example 4:
Review: "Absolutely love it! Best purchase I've made this year."
Sentiment: POSITIVE

Example 5:
Review: "Would not recommend. Cheaply made and overpriced."
Sentiment: NEGATIVE

Now classify this review:
Review: "{review}"
Sentiment:"""


# -----------------------------------------------------------------------------
# 3. CHAIN-OF-THOUGHT PROMPTING (Original - has parsing issues)
# -----------------------------------------------------------------------------
# Ask the model to reason through the problem step-by-step.
# Often improves accuracy on complex reasoning tasks.
#
# PROBLEM: This version doesn't constrain the output format, so the model
# mentions "positive words" and "negative words" in its analysis, which
# confuses our simple keyword-based parser.

CHAIN_OF_THOUGHT_PROMPT = """Classify the sentiment of the following product review as POSITIVE, NEGATIVE, or NEUTRAL.

Think through this step-by-step:
1. Identify key positive words or phrases
2. Identify key negative words or phrases
3. Consider the overall tone and context
4. Make your final classification

Review: "{review}"

Analysis:"""


# -----------------------------------------------------------------------------
# 4. CHAIN-OF-THOUGHT IMPROVED (Fixed version)
# -----------------------------------------------------------------------------
# Key improvements:
# 1. Structured output format with clear "FINAL:" marker
# 2. Avoid asking model to list "positive/negative words" (confuses parser)
# 3. Explicit instruction to end with exactly one word
#
# This demonstrates a core prompt engineering lesson:
# Your prompt must account for how you'll parse the response.

CHAIN_OF_THOUGHT_IMPROVED_PROMPT = """Classify the sentiment of the following product review as POSITIVE, NEGATIVE, or NEUTRAL.

Think through this step-by-step:
1. What is the reviewer's overall experience?
2. What emotions or opinions are expressed?
3. Is the tone favorable, unfavorable, or mixed/neutral?

After your analysis, you MUST end with exactly this format:
FINAL: [your one-word classification]

Review: "{review}"

Analysis:"""


# System prompts for each approach
SYSTEM_PROMPTS = {
    "zero-shot": "You are a sentiment analysis expert. Respond concisely with only the classification.",
    "few-shot": "You are a sentiment analysis expert. Follow the pattern shown in the examples.",
    "chain-of-thought": "You are a sentiment analysis expert. Think carefully and explain your reasoning before giving the final classification.",
    "chain-of-thought-improved": "You are a sentiment analysis expert. Analyze the review step-by-step, then provide your final classification in the exact format requested.",
}


# =============================================================================
# TEST DATASET
# =============================================================================
# Reviews with ground truth labels for evaluation

TEST_REVIEWS = [
    {
        "review": "Amazing product! Works perfectly and arrived ahead of schedule.",
        "label": "POSITIVE",
    },
    {
        "review": "Complete waste of money. Broke after one day of use.",
        "label": "NEGATIVE",
    },
    {
        "review": "It's okay. Does what it says but nothing impressive.",
        "label": "NEUTRAL",
    },
    {
        "review": "Five stars! My whole family loves this. Will buy again.",
        "label": "POSITIVE",
    },
    {
        "review": "Disappointed. The quality doesn't match the price at all.",
        "label": "NEGATIVE",
    },
    {
        "review": "Decent product for the price. Shipping was slow though.",
        "label": "NEUTRAL",
    },
    {
        "review": "This changed my life! Can't believe I waited so long to buy it.",
        "label": "POSITIVE",
    },
    {
        "review": "Awful. Don't buy. Customer support is non-existent.",
        "label": "NEGATIVE",
    },
    {
        "review": "Average quality. Works fine for basic needs.",
        "label": "NEUTRAL",
    },
    {
        "review": "Exceeded all my expectations. Premium quality!",
        "label": "POSITIVE",
    },
]


def format_prompt(template: str, review: str) -> str:
    """Format a prompt template with the given review."""
    return template.format(review=review)


def extract_sentiment(response: str, approach: str) -> str:
    """
    Extract the sentiment label from an LLM response.

    For chain-of-thought, we look for the final classification.
    For others, we look for POSITIVE, NEGATIVE, or NEUTRAL.
    """
    response_upper = response.upper()

    # For improved chain-of-thought, look for "FINAL:" marker
    if approach == "chain-of-thought-improved":
        # Look for FINAL: marker first
        if "FINAL:" in response_upper:
            final_part = response_upper.split("FINAL:")[-1].strip()
            if "POSITIVE" in final_part:
                return "POSITIVE"
            elif "NEGATIVE" in final_part:
                return "NEGATIVE"
            elif "NEUTRAL" in final_part:
                return "NEUTRAL"
        # Fallback: check last line
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        if lines:
            last_line = lines[-1].upper()
            if "POSITIVE" in last_line:
                return "POSITIVE"
            elif "NEGATIVE" in last_line:
                return "NEGATIVE"
            elif "NEUTRAL" in last_line:
                return "NEUTRAL"

    # For original chain-of-thought, look at the end of the response
    if approach == "chain-of-thought":
        # Look for final classification patterns (scan from end)
        lines = response.split('\n')
        for line in reversed(lines):
            line_upper = line.upper()
            if "POSITIVE" in line_upper:
                return "POSITIVE"
            elif "NEGATIVE" in line_upper:
                return "NEGATIVE"
            elif "NEUTRAL" in line_upper:
                return "NEUTRAL"

    # Default: look for keywords anywhere (works for zero-shot, few-shot)
    if "POSITIVE" in response_upper:
        return "POSITIVE"
    elif "NEGATIVE" in response_upper:
        return "NEGATIVE"
    elif "NEUTRAL" in response_upper:
        return "NEUTRAL"

    return "UNKNOWN"
