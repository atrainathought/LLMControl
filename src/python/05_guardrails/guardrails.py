"""
Guardrails Pipeline

This module combines validators into a complete guardrails system:
1. Pre-processing guardrails (before LLM call)
2. Post-processing guardrails (after LLM call)
3. Configurable actions (block, sanitize, warn)
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from validators import (
    ValidationResult,
    ValidationAction,
    PIIDetector,
    PromptInjectionDetector,
    ContentModerator,
    OutputTopicValidator,
    OutputLengthValidator,
    HallucinationDetector,
)


@dataclass
class GuardrailsResult:
    """Result of running guardrails pipeline."""
    original_input: str
    processed_input: str
    original_output: str
    processed_output: str
    blocked: bool = False
    block_reason: Optional[str] = None
    input_validations: List[ValidationResult] = field(default_factory=list)
    output_validations: List[ValidationResult] = field(default_factory=list)
    latency_ms: float = 0


class GuardrailsPipeline:
    """
    Complete guardrails pipeline for LLM applications.

    Usage:
        pipeline = GuardrailsPipeline()
        result = pipeline.run(user_input, llm_call_function)
    """

    def __init__(
        self,
        input_validators: List = None,
        output_validators: List = None,
        verbose: bool = True
    ):
        # Default input validators
        self.input_validators = input_validators or [
            PIIDetector(action=ValidationAction.SANITIZE),
            PromptInjectionDetector(action=ValidationAction.BLOCK),
            ContentModerator(action=ValidationAction.BLOCK),
        ]

        # Default output validators
        self.output_validators = output_validators or [
            OutputLengthValidator(max_length=5000, action=ValidationAction.WARN),
        ]

        self.verbose = verbose

    def validate_input(self, text: str) -> tuple[str, List[ValidationResult], bool, Optional[str]]:
        """
        Run input validators.

        Returns:
            - processed_text: sanitized/modified text
            - validations: list of validation results
            - blocked: whether input was blocked
            - block_reason: reason for blocking (if blocked)
        """
        processed = text
        validations = []
        blocked = False
        block_reason = None

        for validator in self.input_validators:
            result = validator.validate(processed)
            validations.append(result)

            if not result.passed:
                if result.action == ValidationAction.BLOCK:
                    blocked = True
                    block_reason = result.reason
                    if self.verbose:
                        print(f"    [BLOCKED] {result.validator_name}: {result.reason}")
                    break
                elif result.action == ValidationAction.SANITIZE and result.modified_content:
                    processed = result.modified_content
                    if self.verbose:
                        print(f"    [SANITIZED] {result.validator_name}: {result.reason}")
                elif result.action == ValidationAction.WARN:
                    if self.verbose:
                        print(f"    [WARNING] {result.validator_name}: {result.reason}")
            else:
                if self.verbose:
                    print(f"    [PASS] {result.validator_name}")

        return processed, validations, blocked, block_reason

    def validate_output(self, text: str, context: str = None) -> tuple[str, List[ValidationResult]]:
        """
        Run output validators.

        Returns:
            - processed_text: sanitized/modified text
            - validations: list of validation results
        """
        processed = text
        validations = []

        for validator in self.output_validators:
            # Some validators need context
            if hasattr(validator, 'validate') and 'context' in validator.validate.__code__.co_varnames:
                result = validator.validate(processed, context=context)
            else:
                result = validator.validate(processed)

            validations.append(result)

            if not result.passed:
                if result.action == ValidationAction.SANITIZE and result.modified_content:
                    processed = result.modified_content
                    if self.verbose:
                        print(f"    [SANITIZED] {result.validator_name}: {result.reason}")
                elif result.action == ValidationAction.WARN:
                    if self.verbose:
                        print(f"    [WARNING] {result.validator_name}: {result.reason}")
            else:
                if self.verbose:
                    print(f"    [PASS] {result.validator_name}")

        return processed, validations

    def run(
        self,
        user_input: str,
        llm_function,  # Callable that takes str and returns str
        context: str = None
    ) -> GuardrailsResult:
        """
        Run the complete guardrails pipeline.

        Args:
            user_input: User's input text
            llm_function: Function to call LLM (takes processed input, returns output)
            context: Optional context for output validation

        Returns:
            GuardrailsResult with all validation details
        """
        start_time = time.perf_counter()

        result = GuardrailsResult(
            original_input=user_input,
            processed_input="",
            original_output="",
            processed_output=""
        )

        # Step 1: Validate input
        if self.verbose:
            print("\n  Input Validation:")

        processed_input, input_validations, blocked, block_reason = self.validate_input(user_input)
        result.processed_input = processed_input
        result.input_validations = input_validations

        if blocked:
            result.blocked = True
            result.block_reason = block_reason
            result.latency_ms = (time.perf_counter() - start_time) * 1000
            return result

        # Step 2: Call LLM
        if self.verbose:
            print("\n  Calling LLM...")

        llm_output = llm_function(processed_input)
        result.original_output = llm_output

        # Step 3: Validate output
        if self.verbose:
            print("\n  Output Validation:")

        processed_output, output_validations = self.validate_output(llm_output, context)
        result.processed_output = processed_output
        result.output_validations = output_validations

        result.latency_ms = (time.perf_counter() - start_time) * 1000

        return result


# =============================================================================
# PRE-BUILT GUARDRAILS CONFIGURATIONS
# =============================================================================

def create_strict_guardrails() -> GuardrailsPipeline:
    """Create a strict guardrails configuration."""
    return GuardrailsPipeline(
        input_validators=[
            PIIDetector(action=ValidationAction.BLOCK),  # Block any PII
            PromptInjectionDetector(action=ValidationAction.BLOCK),
            ContentModerator(action=ValidationAction.BLOCK),
        ],
        output_validators=[
            OutputLengthValidator(max_length=2000, action=ValidationAction.SANITIZE),
            HallucinationDetector(action=ValidationAction.WARN),
        ]
    )


def create_permissive_guardrails() -> GuardrailsPipeline:
    """Create a more permissive guardrails configuration."""
    return GuardrailsPipeline(
        input_validators=[
            PIIDetector(action=ValidationAction.SANITIZE),  # Just sanitize PII
            PromptInjectionDetector(action=ValidationAction.WARN),  # Warn but allow
        ],
        output_validators=[
            OutputLengthValidator(max_length=10000, action=ValidationAction.WARN),
        ]
    )


def create_customer_support_guardrails(allowed_topics: List[str]) -> GuardrailsPipeline:
    """Create guardrails for a customer support chatbot."""
    return GuardrailsPipeline(
        input_validators=[
            PIIDetector(action=ValidationAction.SANITIZE),
            PromptInjectionDetector(action=ValidationAction.BLOCK),
        ],
        output_validators=[
            OutputTopicValidator(allowed_topics=allowed_topics, action=ValidationAction.WARN),
            OutputLengthValidator(min_length=10, max_length=1000, action=ValidationAction.WARN),
        ]
    )
