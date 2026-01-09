"""
Sample Test Suites for Evaluation.

Provides reusable test cases for different evaluation scenarios:
- QA evaluation (factual questions)
- RAG evaluation (with context)
- Classification evaluation
- Generation quality evaluation
"""

from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class TestCase:
    """A single test case for evaluation."""
    id: str
    input: str  # The prompt/question
    expected: Any = None  # Expected output (format depends on evaluator)
    context: str = None  # Optional context (for RAG evaluations)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# QA TEST CASES - Factual Questions
# =============================================================================

QA_TEST_CASES: List[TestCase] = [
    TestCase(
        id="qa_001",
        input="What is the capital of France?",
        expected="Paris",
        metadata={"category": "geography", "difficulty": "easy"}
    ),
    TestCase(
        id="qa_002",
        input="Who wrote 'Romeo and Juliet'?",
        expected="William Shakespeare",
        metadata={"category": "literature", "difficulty": "easy"}
    ),
    TestCase(
        id="qa_003",
        input="What is the chemical symbol for gold?",
        expected="Au",
        metadata={"category": "science", "difficulty": "easy"}
    ),
    TestCase(
        id="qa_004",
        input="In what year did World War II end?",
        expected="1945",
        metadata={"category": "history", "difficulty": "easy"}
    ),
    TestCase(
        id="qa_005",
        input="What is the largest planet in our solar system?",
        expected="Jupiter",
        metadata={"category": "science", "difficulty": "easy"}
    ),
    TestCase(
        id="qa_006",
        input="What programming language was created by Guido van Rossum?",
        expected="Python",
        metadata={"category": "technology", "difficulty": "medium"}
    ),
    TestCase(
        id="qa_007",
        input="What is the speed of light in meters per second?",
        expected="299,792,458 meters per second",
        metadata={"category": "physics", "difficulty": "medium"}
    ),
    TestCase(
        id="qa_008",
        input="Who painted the Mona Lisa?",
        expected="Leonardo da Vinci",
        metadata={"category": "art", "difficulty": "easy"}
    ),
]


# =============================================================================
# RAG TEST CASES - Questions with Context
# =============================================================================

RAG_TEST_CASES: List[TestCase] = [
    TestCase(
        id="rag_001",
        input="What is the company's refund policy?",
        expected="30-day money-back guarantee for unused items",
        context="""
        COMPANY POLICIES

        Return Policy:
        We offer a 30-day money-back guarantee for all unused items in original packaging.
        Items must be returned within 30 days of purchase for a full refund.
        Used or damaged items are not eligible for returns.

        Shipping Policy:
        Free shipping on orders over $50. Standard delivery takes 3-5 business days.
        Express shipping available for $9.99.
        """,
        metadata={"category": "policy", "doc_type": "internal"}
    ),
    TestCase(
        id="rag_002",
        input="How many employees does the company have?",
        expected="500 employees",
        context="""
        COMPANY OVERVIEW

        Founded in 2010, TechCorp has grown to become a leader in cloud solutions.
        The company currently employs approximately 500 people across 5 offices.
        Headquarters is located in San Francisco, California.
        Annual revenue exceeded $100 million in 2023.
        """,
        metadata={"category": "company_info", "doc_type": "internal"}
    ),
    TestCase(
        id="rag_003",
        input="What are the system requirements for the software?",
        expected="Windows 10 or later, 8GB RAM, 500MB disk space",
        context="""
        SOFTWARE REQUIREMENTS

        Minimum System Requirements:
        - Operating System: Windows 10 or later, macOS 10.15+, or Ubuntu 20.04+
        - RAM: 8GB minimum, 16GB recommended
        - Storage: 500MB available disk space
        - Internet: Broadband connection required for cloud features

        Supported browsers: Chrome, Firefox, Safari, Edge (latest versions)
        """,
        metadata={"category": "technical", "doc_type": "documentation"}
    ),
    TestCase(
        id="rag_004",
        input="What programming languages does the API support?",
        expected="Python, JavaScript, Java, and Go",
        context="""
        API DOCUMENTATION

        Getting Started:
        Our REST API provides programmatic access to all platform features.

        Official SDKs are available for:
        - Python (3.8+)
        - JavaScript/Node.js (14+)
        - Java (11+)
        - Go (1.18+)

        Authentication uses OAuth 2.0 with JWT tokens.
        Rate limits: 1000 requests per hour for free tier, unlimited for enterprise.
        """,
        metadata={"category": "api", "doc_type": "documentation"}
    ),
]


# =============================================================================
# CLASSIFICATION TEST CASES
# =============================================================================

CLASSIFICATION_TEST_CASES: List[TestCase] = [
    TestCase(
        id="cls_001",
        input="I love this product! Best purchase I've ever made.",
        expected="positive",
        metadata={"category": "sentiment"}
    ),
    TestCase(
        id="cls_002",
        input="Terrible experience. The product broke after one day.",
        expected="negative",
        metadata={"category": "sentiment"}
    ),
    TestCase(
        id="cls_003",
        input="It's okay, nothing special but does the job.",
        expected="neutral",
        metadata={"category": "sentiment"}
    ),
    TestCase(
        id="cls_004",
        input="My API calls are returning 500 errors",
        expected="technical",
        metadata={"category": "support_ticket"}
    ),
    TestCase(
        id="cls_005",
        input="I was charged twice for my subscription",
        expected="billing",
        metadata={"category": "support_ticket"}
    ),
    TestCase(
        id="cls_006",
        input="What enterprise plans do you offer?",
        expected="sales",
        metadata={"category": "support_ticket"}
    ),
]


# =============================================================================
# CODE GENERATION TEST CASES
# =============================================================================

CODE_GENERATION_TEST_CASES: List[TestCase] = [
    TestCase(
        id="code_001",
        input="Write a Python function that reverses a string",
        expected=["def", "return", "reverse", "[::-1]"],  # Keywords to check
        metadata={"language": "python", "difficulty": "easy"}
    ),
    TestCase(
        id="code_002",
        input="Write a Python function to check if a number is prime",
        expected=["def", "return", "True", "False", "for", "range"],
        metadata={"language": "python", "difficulty": "medium"}
    ),
    TestCase(
        id="code_003",
        input="Write a JavaScript function that finds the maximum value in an array",
        expected=["function", "return", "Math.max", "array"],
        metadata={"language": "javascript", "difficulty": "easy"}
    ),
]


# =============================================================================
# SUMMARIZATION TEST CASES
# =============================================================================

SUMMARIZATION_TEST_CASES: List[TestCase] = [
    TestCase(
        id="sum_001",
        input="Summarize the following article in 2-3 sentences",
        expected=None,  # Will use LLM judge
        context="""
        Artificial intelligence has made remarkable strides in recent years, transforming
        industries from healthcare to finance. Machine learning models can now diagnose
        diseases with accuracy rivaling human doctors, while AI-powered trading algorithms
        manage billions in assets. However, experts warn that these advances come with
        significant ethical concerns, including job displacement, privacy issues, and the
        potential for algorithmic bias. Regulators worldwide are scrambling to create
        frameworks that balance innovation with public safety. Despite the challenges,
        investment in AI continues to surge, with global spending expected to exceed
        $500 billion by 2025.
        """,
        metadata={"max_words": 50, "style": "neutral"}
    ),
    TestCase(
        id="sum_002",
        input="Provide a one-sentence summary of the key point",
        expected=None,
        context="""
        The quarterly earnings report shows that revenue increased by 25% year-over-year,
        driven primarily by strong growth in the cloud services division. Operating expenses
        remained flat, resulting in a 40% improvement in operating margin. The company
        raised full-year guidance and announced a $1 billion share buyback program.
        """,
        metadata={"max_words": 25, "style": "business"}
    ),
]


# =============================================================================
# TEST SUITE FACTORY
# =============================================================================

def get_test_suite(suite_name: str) -> List[TestCase]:
    """
    Get a predefined test suite by name.

    Available suites:
    - qa: Factual Q&A questions
    - rag: RAG questions with context
    - classification: Classification tasks
    - code: Code generation
    - summarization: Text summarization
    """
    suites = {
        "qa": QA_TEST_CASES,
        "rag": RAG_TEST_CASES,
        "classification": CLASSIFICATION_TEST_CASES,
        "code": CODE_GENERATION_TEST_CASES,
        "summarization": SUMMARIZATION_TEST_CASES,
    }

    if suite_name not in suites:
        raise ValueError(f"Unknown suite: {suite_name}. Available: {list(suites.keys())}")

    return suites[suite_name]


def create_custom_test_suite(
    questions: List[str],
    expected_answers: List[str] = None,
    contexts: List[str] = None,
    prefix: str = "custom",
) -> List[TestCase]:
    """
    Create a custom test suite from lists.

    Args:
        questions: List of questions/prompts
        expected_answers: Optional list of expected answers
        contexts: Optional list of contexts (for RAG)
        prefix: Prefix for test case IDs

    Returns:
        List of TestCase objects
    """
    test_cases = []

    for i, question in enumerate(questions):
        expected = expected_answers[i] if expected_answers and i < len(expected_answers) else None
        context = contexts[i] if contexts and i < len(contexts) else None

        test_cases.append(TestCase(
            id=f"{prefix}_{i:03d}",
            input=question,
            expected=expected,
            context=context,
        ))

    return test_cases


# =============================================================================
# BENCHMARK DATASETS (for more rigorous testing)
# =============================================================================

def create_mmlu_sample() -> List[TestCase]:
    """
    Sample from MMLU-style questions (multiple choice knowledge).
    """
    return [
        TestCase(
            id="mmlu_001",
            input="""What is the primary function of mitochondria in a cell?
A) Protein synthesis
B) Energy production (ATP)
C) Cell division
D) Waste removal

Answer with just the letter.""",
            expected="B",
            metadata={"subject": "biology", "source": "mmlu_sample"}
        ),
        TestCase(
            id="mmlu_002",
            input="""In economics, what does GDP stand for?
A) Gross Domestic Product
B) General Development Plan
C) Global Distribution Program
D) Government Debt Portfolio

Answer with just the letter.""",
            expected="A",
            metadata={"subject": "economics", "source": "mmlu_sample"}
        ),
        TestCase(
            id="mmlu_003",
            input="""Which sorting algorithm has the best average-case time complexity?
A) Bubble Sort - O(n²)
B) Quick Sort - O(n log n)
C) Selection Sort - O(n²)
D) Insertion Sort - O(n²)

Answer with just the letter.""",
            expected="B",
            metadata={"subject": "computer_science", "source": "mmlu_sample"}
        ),
    ]


def create_truthfulqa_sample() -> List[TestCase]:
    """
    Sample from TruthfulQA-style questions (checking for hallucinations).
    """
    return [
        TestCase(
            id="truthful_001",
            input="What happens if you crack your knuckles?",
            expected="Cracking knuckles does not cause arthritis",  # Common myth
            metadata={"category": "health_myths", "source": "truthfulqa_sample"}
        ),
        TestCase(
            id="truthful_002",
            input="Do we only use 10% of our brain?",
            expected="No, this is a myth. We use virtually all of our brain.",
            metadata={"category": "neuroscience_myths", "source": "truthfulqa_sample"}
        ),
        TestCase(
            id="truthful_003",
            input="What is the Great Wall of China visible from?",
            expected="The Great Wall is not visible from space with the naked eye",
            metadata={"category": "common_misconceptions", "source": "truthfulqa_sample"}
        ),
    ]
