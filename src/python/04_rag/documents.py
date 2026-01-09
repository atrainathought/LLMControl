"""
Document Processing for RAG

This module handles:
1. Sample knowledge base (fictional company docs)
2. Document chunking strategies
3. Metadata extraction
"""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class Document:
    """A document with content and metadata."""
    id: str
    content: str
    metadata: Dict[str, Any]


@dataclass
class Chunk:
    """A chunk of a document for embedding."""
    id: str
    content: str
    doc_id: str
    metadata: Dict[str, Any]


# =============================================================================
# SAMPLE KNOWLEDGE BASE
# Fictional company "TechFlow Inc." documentation
# =============================================================================

KNOWLEDGE_BASE = [
    Document(
        id="doc_001",
        content="""
# TechFlow Inc. Employee Handbook - Leave Policy

## Annual Leave
All full-time employees are entitled to 20 days of paid annual leave per year.
Leave accrues at 1.67 days per month. New employees can start using leave after
completing their 3-month probation period.

## Sick Leave
Employees receive 10 days of paid sick leave per year. For absences longer than
3 consecutive days, a medical certificate is required. Unused sick leave does not
carry over to the next year.

## Parental Leave
- Primary caregiver: 16 weeks paid leave
- Secondary caregiver: 4 weeks paid leave
Parental leave must be taken within 12 months of the child's birth or adoption.

## Public Holidays
TechFlow observes 10 public holidays per year. If a public holiday falls on a
weekend, the following Monday is observed as a day off.

## Leave Request Process
1. Submit leave request through the HR portal at least 2 weeks in advance
2. Manager approval required for leaves longer than 3 days
3. Emergency leave can be requested retroactively within 48 hours
""",
        metadata={"category": "hr", "topic": "leave_policy", "last_updated": "2024-01-15"}
    ),

    Document(
        id="doc_002",
        content="""
# TechFlow Inc. IT Security Policy

## Password Requirements
- Minimum 12 characters
- Must include: uppercase, lowercase, number, and special character
- Passwords expire every 90 days
- Cannot reuse last 10 passwords
- Multi-factor authentication (MFA) is mandatory for all accounts

## Data Classification
1. Public: Marketing materials, public website content
2. Internal: Internal memos, non-sensitive business data
3. Confidential: Customer data, financial records, employee PII
4. Restricted: Trade secrets, security credentials, encryption keys

## Remote Work Security
- Use company VPN for all work-related activities
- Do not use public WiFi without VPN
- Lock your screen when away from your device
- Report lost or stolen devices immediately to IT Security

## Incident Reporting
Security incidents must be reported within 1 hour to security@techflow.com.
Include: date/time, description, affected systems, and any actions taken.

## Software Installation
Only IT-approved software may be installed on company devices. Request new
software through the IT portal. Unapproved software will be flagged and removed.
""",
        metadata={"category": "it", "topic": "security", "last_updated": "2024-02-01"}
    ),

    Document(
        id="doc_003",
        content="""
# TechFlow Inc. Expense Reimbursement Policy

## Eligible Expenses
- Business travel (flights, hotels, ground transportation)
- Client meals and entertainment (pre-approval required over $100)
- Professional development and training
- Home office equipment (up to $500 annually)
- Mobile phone plan (up to $75/month)

## Expense Limits
| Category | Daily Limit | Approval Required |
|----------|-------------|-------------------|
| Meals (domestic) | $75 | No |
| Meals (international) | $100 | No |
| Hotels | $250 | Manager for >$250 |
| Flights | Economy class | Manager for business class |

## Submission Process
1. Submit expenses within 30 days of incurring them
2. Attach original receipts (photos acceptable)
3. Include business justification for each expense
4. Expenses over $500 require manager pre-approval

## Reimbursement Timeline
- Standard processing: 5-7 business days
- Expenses submitted by the 15th are paid with that month's payroll
- International expenses may take up to 14 days

## Non-Reimbursable Items
- Personal travel upgrades
- Alcohol (except at approved client events)
- Traffic violations or parking tickets
- Personal phone calls or subscriptions
""",
        metadata={"category": "finance", "topic": "expenses", "last_updated": "2024-01-20"}
    ),

    Document(
        id="doc_004",
        content="""
# TechFlow Inc. Product: DataSync Pro

## Overview
DataSync Pro is TechFlow's flagship data integration platform. It enables
real-time synchronization between cloud services, databases, and enterprise
applications.

## Key Features
- Real-time bidirectional sync
- Support for 200+ connectors (Salesforce, SAP, Oracle, etc.)
- No-code transformation pipeline
- Enterprise-grade security (SOC 2, HIPAA, GDPR compliant)
- 99.99% uptime SLA

## Pricing Tiers
1. Starter: $99/month - 5 connectors, 100k records/month
2. Professional: $499/month - 25 connectors, 1M records/month
3. Enterprise: Custom pricing - Unlimited connectors, dedicated support

## Technical Specifications
- API rate limit: 1000 requests/minute (Pro), 5000 (Enterprise)
- Maximum record size: 10MB
- Supported databases: PostgreSQL, MySQL, MongoDB, Oracle, SQL Server
- Deployment options: Cloud (AWS, Azure, GCP) or On-premise

## Support
- Starter: Email support, 48-hour response time
- Professional: Email + chat, 24-hour response time
- Enterprise: 24/7 phone support, dedicated account manager
""",
        metadata={"category": "product", "topic": "datasync_pro", "last_updated": "2024-03-01"}
    ),

    Document(
        id="doc_005",
        content="""
# TechFlow Inc. Engineering On-Call Policy

## On-Call Rotation
- Each team maintains a weekly on-call rotation
- On-call shift: Monday 9 AM to Monday 9 AM (7 days)
- Primary and secondary on-call assigned each week
- Minimum 4 people per rotation to ensure adequate rest

## Response Time Requirements
| Severity | Response Time | Resolution Target |
|----------|---------------|-------------------|
| SEV1 (Critical) | 15 minutes | 4 hours |
| SEV2 (High) | 30 minutes | 8 hours |
| SEV3 (Medium) | 2 hours | 24 hours |
| SEV4 (Low) | Next business day | Best effort |

## On-Call Compensation
- Weekday on-call: $200/day stipend
- Weekend/holiday on-call: $300/day stipend
- Incident response: 1.5x hourly rate for time worked
- Compensatory time off: 1 day off for each weekend worked

## Escalation Path
1. Primary on-call engineer
2. Secondary on-call engineer (after 15 min no response)
3. Team lead (after 30 min no response)
4. Engineering manager (for SEV1 incidents)
5. VP of Engineering (for customer-impacting SEV1)

## Handoff Procedure
- Review open incidents and ongoing issues
- Update on-call documentation
- Test alerting systems (PagerDuty)
- Confirm contact information is current
""",
        metadata={"category": "engineering", "topic": "oncall", "last_updated": "2024-02-15"}
    ),
]


# =============================================================================
# TEST QUESTIONS (with ground truth answers from the docs)
# =============================================================================

TEST_QUESTIONS = [
    {
        "question": "How many days of annual leave do employees get at TechFlow?",
        "answer": "20 days per year",
        "source_doc": "doc_001",
        "category": "hr"
    },
    {
        "question": "What is the password expiration policy at TechFlow?",
        "answer": "Passwords expire every 90 days",
        "source_doc": "doc_002",
        "category": "it"
    },
    {
        "question": "What is the daily meal expense limit for domestic travel?",
        "answer": "$75 per day",
        "source_doc": "doc_003",
        "category": "finance"
    },
    {
        "question": "What is the uptime SLA for DataSync Pro?",
        "answer": "99.99% uptime SLA",
        "source_doc": "doc_004",
        "category": "product"
    },
    {
        "question": "How much is the on-call stipend for weekends at TechFlow?",
        "answer": "$300/day",
        "source_doc": "doc_005",
        "category": "engineering"
    },
    {
        "question": "How long is parental leave for the primary caregiver?",
        "answer": "16 weeks paid leave",
        "source_doc": "doc_001",
        "category": "hr"
    },
    {
        "question": "What is the response time for SEV1 incidents?",
        "answer": "15 minutes",
        "source_doc": "doc_005",
        "category": "engineering"
    },
    {
        "question": "How many connectors are included in the Professional tier of DataSync Pro?",
        "answer": "25 connectors",
        "source_doc": "doc_004",
        "category": "product"
    },
]


# =============================================================================
# CHUNKING STRATEGIES
# =============================================================================

def chunk_by_paragraphs(doc: Document, max_chunk_size: int = 500) -> List[Chunk]:
    """
    Split document into chunks by paragraphs.

    Simple strategy: split on double newlines, combine small paragraphs.
    """
    paragraphs = [p.strip() for p in doc.content.split('\n\n') if p.strip()]

    chunks = []
    current_chunk = ""
    chunk_idx = 0

    for para in paragraphs:
        if len(current_chunk) + len(para) < max_chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(Chunk(
                    id=f"{doc.id}_chunk_{chunk_idx}",
                    content=current_chunk.strip(),
                    doc_id=doc.id,
                    metadata={**doc.metadata, "chunk_index": chunk_idx}
                ))
                chunk_idx += 1
            current_chunk = para + "\n\n"

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(Chunk(
            id=f"{doc.id}_chunk_{chunk_idx}",
            content=current_chunk.strip(),
            doc_id=doc.id,
            metadata={**doc.metadata, "chunk_index": chunk_idx}
        ))

    return chunks


def chunk_by_headers(doc: Document) -> List[Chunk]:
    """
    Split document by markdown headers.

    Better for structured documents - each section becomes a chunk.
    """
    lines = doc.content.split('\n')
    chunks = []
    current_chunk = ""
    current_header = ""
    chunk_idx = 0

    for line in lines:
        if line.startswith('## '):
            # New section - save previous chunk
            if current_chunk.strip():
                chunks.append(Chunk(
                    id=f"{doc.id}_chunk_{chunk_idx}",
                    content=current_chunk.strip(),
                    doc_id=doc.id,
                    metadata={**doc.metadata, "chunk_index": chunk_idx, "section": current_header}
                ))
                chunk_idx += 1
            current_header = line.replace('## ', '')
            current_chunk = line + '\n'
        else:
            current_chunk += line + '\n'

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(Chunk(
            id=f"{doc.id}_chunk_{chunk_idx}",
            content=current_chunk.strip(),
            doc_id=doc.id,
            metadata={**doc.metadata, "chunk_index": chunk_idx, "section": current_header}
        ))

    return chunks


def chunk_all_documents(docs: List[Document], strategy: str = "paragraphs") -> List[Chunk]:
    """Chunk all documents using the specified strategy."""
    all_chunks = []

    for doc in docs:
        if strategy == "paragraphs":
            chunks = chunk_by_paragraphs(doc)
        elif strategy == "headers":
            chunks = chunk_by_headers(doc)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        all_chunks.extend(chunks)

    return all_chunks
