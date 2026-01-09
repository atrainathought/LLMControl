"""
Input and Output Validators for Guardrails

This module implements safety checks for LLM applications:
1. INPUT validators - check user input before sending to LLM
2. OUTPUT validators - check LLM response before returning to user

Each validator returns a ValidationResult with:
- passed: bool
- reason: str (if blocked)
- modified_content: str (if sanitized)
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Callable, Dict, Any
from enum import Enum


class ValidationAction(Enum):
    """What to do when validation fails."""
    BLOCK = "block"      # Reject entirely
    SANITIZE = "sanitize"  # Remove/mask sensitive content
    WARN = "warn"        # Allow but log warning
    PASS = "pass"        # Allow through


@dataclass
class ValidationResult:
    """Result of a validation check."""
    passed: bool
    action: ValidationAction
    reason: Optional[str] = None
    original_content: str = ""
    modified_content: Optional[str] = None
    validator_name: str = ""
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


# =============================================================================
# INPUT VALIDATORS
# =============================================================================

class PIIDetector:
    """
    Detect Personally Identifiable Information (PII) in text.

    Detects:
    - Email addresses
    - Phone numbers
    - Social Security Numbers
    - Credit card numbers
    - IP addresses
    """

    PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
        "ssn": r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
        "credit_card": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
    }

    def __init__(self, action: ValidationAction = ValidationAction.SANITIZE):
        self.action = action

    def validate(self, text: str) -> ValidationResult:
        """Check for PII in text."""
        found_pii = {}

        for pii_type, pattern in self.PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                found_pii[pii_type] = matches

        if not found_pii:
            return ValidationResult(
                passed=True,
                action=ValidationAction.PASS,
                original_content=text,
                validator_name="PIIDetector"
            )

        # Sanitize by masking PII
        sanitized = text
        for pii_type, matches in found_pii.items():
            for match in matches:
                mask = f"[{pii_type.upper()}_REDACTED]"
                sanitized = sanitized.replace(match, mask)

        return ValidationResult(
            passed=False,
            action=self.action,
            reason=f"PII detected: {', '.join(found_pii.keys())}",
            original_content=text,
            modified_content=sanitized,
            validator_name="PIIDetector",
            details={"found_pii": found_pii}
        )


class PromptInjectionDetector:
    """
    Detect potential prompt injection attacks.

    Looks for patterns that might try to:
    - Override system instructions
    - Extract system prompts
    - Bypass safety measures
    """

    SUSPICIOUS_PATTERNS = [
        # Instruction override attempts
        (r'ignore (?:all |previous |prior |above )?(?:instructions?|prompts?|rules?)', "instruction_override"),
        (r'disregard (?:all |previous |prior |above )?(?:instructions?|prompts?|rules?)', "instruction_override"),
        (r'forget (?:all |previous |prior |above )?(?:instructions?|prompts?|rules?)', "instruction_override"),
        (r'do not follow (?:the |your )?(?:instructions?|prompts?|rules?)', "instruction_override"),

        # Role manipulation
        (r'you are now (?:a |an )?', "role_manipulation"),
        (r'pretend (?:to be|you\'?re) (?:a |an )?', "role_manipulation"),
        (r'act as (?:a |an |if )?', "role_manipulation"),
        (r'roleplay as', "role_manipulation"),

        # System prompt extraction
        (r'(?:what|show|tell|reveal|display|print|output|repeat) (?:me |us )?(?:your |the )?(?:system |initial )?(?:prompt|instructions?|rules?)', "prompt_extraction"),
        (r'(?:what|how) (?:were|are) you (?:instructed|programmed|told|prompted)', "prompt_extraction"),

        # Jailbreak attempts
        (r'(?:DAN|developer mode|god mode|sudo mode)', "jailbreak"),
        (r'bypass (?:your |the |any )?(?:safety|restrictions?|limitations?|filters?)', "jailbreak"),

        # Delimiter injection
        (r'<\/?(?:system|user|assistant|human|ai)>', "delimiter_injection"),
        (r'\[(?:SYSTEM|INST|\/INST)\]', "delimiter_injection"),
    ]

    def __init__(self, action: ValidationAction = ValidationAction.BLOCK):
        self.action = action

    def validate(self, text: str) -> ValidationResult:
        """Check for prompt injection attempts."""
        detected = []

        text_lower = text.lower()
        for pattern, attack_type in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, text_lower):
                detected.append(attack_type)

        if not detected:
            return ValidationResult(
                passed=True,
                action=ValidationAction.PASS,
                original_content=text,
                validator_name="PromptInjectionDetector"
            )

        return ValidationResult(
            passed=False,
            action=self.action,
            reason=f"Potential prompt injection detected: {', '.join(set(detected))}",
            original_content=text,
            validator_name="PromptInjectionDetector",
            details={"attack_types": list(set(detected))}
        )


class ContentModerator:
    """
    Basic content moderation for inappropriate content.

    This is a simple keyword-based approach. Production systems should
    use ML-based classifiers or external moderation APIs.
    """

    # Simple blocklist (in production, use ML-based detection)
    BLOCKED_PATTERNS = [
        (r'\b(?:hack|exploit|attack)\s+(?:the |a |an )?(?:system|server|database|website)\b', "harmful_instructions"),
        (r'\b(?:make|create|build)\s+(?:a |an )?(?:bomb|weapon|explosive)\b', "dangerous_content"),
        (r'\b(?:how to|ways to)\s+(?:harm|hurt|kill|injure)\b', "violent_content"),
    ]

    def __init__(self, action: ValidationAction = ValidationAction.BLOCK):
        self.action = action

    def validate(self, text: str) -> ValidationResult:
        """Check for inappropriate content."""
        detected = []

        text_lower = text.lower()
        for pattern, content_type in self.BLOCKED_PATTERNS:
            if re.search(pattern, text_lower):
                detected.append(content_type)

        if not detected:
            return ValidationResult(
                passed=True,
                action=ValidationAction.PASS,
                original_content=text,
                validator_name="ContentModerator"
            )

        return ValidationResult(
            passed=False,
            action=self.action,
            reason=f"Inappropriate content detected: {', '.join(set(detected))}",
            original_content=text,
            validator_name="ContentModerator",
            details={"content_types": list(set(detected))}
        )


# =============================================================================
# OUTPUT VALIDATORS
# =============================================================================

class OutputTopicValidator:
    """
    Ensure LLM output stays on topic.

    Validates that responses are relevant to the expected domain.
    """

    def __init__(
        self,
        allowed_topics: List[str],
        action: ValidationAction = ValidationAction.WARN
    ):
        self.allowed_topics = [t.lower() for t in allowed_topics]
        self.action = action

    def validate(self, text: str, context: Dict[str, Any] = None) -> ValidationResult:
        """Check if output is on topic."""
        text_lower = text.lower()

        # Simple keyword-based topic detection
        # In production, use embeddings or classifiers
        topic_scores = {}
        for topic in self.allowed_topics:
            # Count mentions of topic-related words
            score = text_lower.count(topic)
            topic_scores[topic] = score

        max_score = max(topic_scores.values()) if topic_scores else 0

        if max_score > 0:
            return ValidationResult(
                passed=True,
                action=ValidationAction.PASS,
                original_content=text,
                validator_name="OutputTopicValidator",
                details={"topic_scores": topic_scores}
            )

        return ValidationResult(
            passed=False,
            action=self.action,
            reason="Response may be off-topic",
            original_content=text,
            validator_name="OutputTopicValidator",
            details={"topic_scores": topic_scores}
        )


class OutputLengthValidator:
    """
    Ensure output meets length requirements.
    """

    def __init__(
        self,
        min_length: int = 0,
        max_length: int = 10000,
        action: ValidationAction = ValidationAction.WARN
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.action = action

    def validate(self, text: str) -> ValidationResult:
        """Check output length."""
        length = len(text)

        if self.min_length <= length <= self.max_length:
            return ValidationResult(
                passed=True,
                action=ValidationAction.PASS,
                original_content=text,
                validator_name="OutputLengthValidator",
                details={"length": length}
            )

        if length < self.min_length:
            reason = f"Output too short ({length} < {self.min_length} chars)"
        else:
            reason = f"Output too long ({length} > {self.max_length} chars)"

        return ValidationResult(
            passed=False,
            action=self.action,
            reason=reason,
            original_content=text,
            modified_content=text[:self.max_length] if length > self.max_length else None,
            validator_name="OutputLengthValidator",
            details={"length": length}
        )


class HallucinationDetector:
    """
    Basic hallucination detection.

    Checks if the output makes claims that aren't supported by the context.
    This is a simple heuristic - production systems need more sophisticated approaches.
    """

    CONFIDENCE_PHRASES = [
        "I'm not sure",
        "I don't know",
        "I cannot confirm",
        "I don't have information",
        "I'm unable to verify",
    ]

    OVERCONFIDENCE_PHRASES = [
        "definitely",
        "certainly",
        "absolutely",
        "without a doubt",
        "100%",
    ]

    def __init__(self, action: ValidationAction = ValidationAction.WARN):
        self.action = action

    def validate(self, text: str, context: str = None) -> ValidationResult:
        """Check for potential hallucinations."""
        text_lower = text.lower()

        # Check for overconfident claims without context support
        has_overconfidence = any(phrase in text_lower for phrase in self.OVERCONFIDENCE_PHRASES)
        has_uncertainty = any(phrase in text_lower for phrase in self.CONFIDENCE_PHRASES)

        # If we have context, check if claims appear grounded
        context_grounded = True
        if context and has_overconfidence:
            # Simple check: does the response reference the context?
            context_words = set(context.lower().split())
            response_words = set(text_lower.split())
            overlap = len(context_words & response_words)
            context_grounded = overlap > 10  # Arbitrary threshold

        if not has_overconfidence or context_grounded:
            return ValidationResult(
                passed=True,
                action=ValidationAction.PASS,
                original_content=text,
                validator_name="HallucinationDetector"
            )

        return ValidationResult(
            passed=False,
            action=self.action,
            reason="Response may contain ungrounded claims",
            original_content=text,
            validator_name="HallucinationDetector",
            details={
                "has_overconfidence": has_overconfidence,
                "context_grounded": context_grounded
            }
        )
