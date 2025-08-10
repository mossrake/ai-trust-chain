# AI Trust Chain Framework - Project Context

## Overview
This document captures the key design decisions and implementation details for the AI Trust Chain Framework reference implementation. Use this to quickly restore context in new conversations.

## Core Concept
The AI Trust Chain Framework addresses the fundamental challenge of making confident, defensible decisions based on AI recommendations by establishing trust characterization at every decision input and maintaining complete audit trails using immutable ledger systems.

## Key Design Decisions

### 1. Trust Architecture
- **Trust vs Confidence**: Strictly separated dual-channel metadata
  - **Confidence**: Self-reported certainty by the endpoint (0.0-1.0)
  - **Trust**: System-assessed trustworthiness (0.0-1.0)
  - High confidence + Low trust = Highest risk scenario

### 2. Trust Representation
- **Single trust value**: Trust is a single composite value, NOT a multi-dimensional structure
- **Trust explanations**: Optional natural language strings that convey dimensional details
- **Internal dimensions**: Endpoints evaluate dimensions privately (calibration, freshness, etc.)
  - Example: "Trust is low (0.3) due to sensor being uncalibrated for 3 years"


### 3. Common Dimension
- **Temporal validity**: The common dimension exposed (0.0-1.0, where 1.0 = fresh)
- Programmatically accessible without parsing explanations
- Can decay based on age or domain-specific rules
- All other dimensions are internal to endpoints and expressed in explanations

### 4. Authentication/Authorization
- **Pass-through by design**: This is intentional, NOT a limitation
- No credential storage ever - stateless design
- Integrates with existing enterprise auth (OAuth, SAML, JWT, etc.)
- System never stores credentials or permissions
- Admin detection is implementation-specific (example uses "admin" prefix)

### 5. Materiality
- **Internal to endpoints**: Each endpoint determines internally how important inputs are
- Not passed via API or stored in assertions
- Sophisticated endpoints determine dynamically based on context
- Reference implementation uses equal weights for simplicity

### 6. Trust Propagation
- **Centralized authority**: Assigns maximum trust ceilings per endpoint class
- **Propagation methods**: Minimum, weighted average, consensus
- **Trust ceiling constraint**: Trust can never exceed endpoint class maximum
- **Provenance chain**: Tracks path through endpoint classes (factual, not trust dimension)

### 7. Intelligent Endpoints
- **Progressive sophistication**: Simple sensors → ML models → LLMs
- **Explanation interpretation**: Sophisticated endpoints can read and understand trust explanations
- **Trust override capability**: Can adjust trust based on context, bounded by assigned trust ceieling
  - Example: Freezer monitor upgrades trust for "out of range" temperature sensor
- **LLM integration**: For INTERPRETATION, not required for generation
  - Simple templates can generate explanations
  - AI (LLMs) may be used to understand and act on explanations

### 8. Blockchain Implementation
- **SQLite for development**: Simple, file-based blockchain
- **PostgreSQL for production**: Planned upgrade path
- **Immutable audit trail**: Complete provenance with all metadata
- **Simple proof-of-work**: Basic mining, not production consensus

### 9. Evidence Chain ("Show Me The Evidence")
- **Complete tracing**: Any assertion can be traced to source
- **Weakest link analysis**: Automatically identifies problematic inputs
- **Multiple formats**: JSON (programmatic), text/markdown (human)
- **Trust-confidence matrix**: Decision recommendations at each level

## Implementation Status

### ✅ Fully Implemented (reference)
- Core trust kernel with propagation
- REST API with all CRUD operations
- Blockchain audit trail (SQLite)
- Evidence chain tracing
- Trust-confidence matrix
- Pass-through authentication
- Materiality weights
- Trust explanations (optional)
- Temporal validity tracking
- "Show me the evidence" functionality

### 🔴 Stubbed (Comments in Code)
- LLM service integration for interpretation
- Trust override logic based on explanations
- Dynamic materiality calculation
- Temporal decay functions
- Explanation generation with LLMs

### 📝 Not Yet Implemented (TODO)
- **HIGH PRIORITY**: Stakeholder Portal UI
- PostgreSQL adapter
- Docker containerization
- Comprehensive test suite
- API documentation (OpenAPI/Swagger)

## File Structure
```
ai-trust-chain/
├── trust_core_kernel.py      # Core trust engine with blockchain
├── trust_rest_api.py         # Flask REST API server
├── README.md           # Complete documentation
├── LICENSE             # AGPL-3.0 with IP notice
├── requirements.txt    # Python dependencies
├── .gitignore         # Git ignore patterns
└── PROJECT_CONTEXT.md  # This file
```

## API Patterns

### Creating Assertion
```python
POST /api/v1/assertions
{
  "endpoint_id": "ml-model-01",
  "endpoint_class": "ml_model.predictive",
  "endpoint_type": "ml_model",
  "content": {"prediction": "failure_likely"},
  "confidence": 0.82,
  "consumed_assertions": ["sensor-001", "sensor-002"]
}

# Note: The endpoint internally decides that sensor-001 has 0.8 materiality
# and sensor-002 has 0.2 materiality based on its logic
```

### Response Includes
```json
{
  "assertion_id": "ml-001",
  "trust_value": 0.72,
  "confidence_value": 0.82,
  "temporal_validity": 0.95,
  "trust_explanation": "Trust is moderate due to..."
}
```

### Getting Matrix Classification (Separate Call)
```python
GET /api/v1/assertions/ml-001/matrix

Response:
{
  "recommendation": "Proceed with minimal oversight",
  "trust_value": 0.72,
  "confidence_value": 0.82
}
```

## Intellectual Property Notice
- **Framework Design**: Proprietary IP of Mossrake Group, LLC
- **Implementation Code**: Released under AGPL-3.0
- **Whitepaper**: Proprietary documentation
- Using the code does not grant rights to the framework design beyond what's needed for AGPL compliance

## Design Philosophy

### What This System IS
- A trust and confidence tracking system
- An immutable audit trail for AI decisions
- A framework for explainable AI
- Uses a pass-through authentication system
- A reference implementation of the Mossrake whitepaper "AI Trust Chain" Version 1.0 https://mossrake.com/ai-trust-chain

### What This System IS NOT
- An authentication/authorization system (delegates to external)
- A credential store (stateless by design)
- An AI/ML platform (integrates with existing)
- A complete production system (reference implementation)

## Quick Start for Development
```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
python rest_api.py

# Initialize system (admin token)
curl -X POST http://localhost:5000/api/v1/admin/initialize \
  -H "Authorization: Bearer admin-token"

# Create assertion
curl -X POST http://localhost:5000/api/v1/assertions \
  -H "Authorization: Bearer user-token" \
  -H "Content-Type: application/json" \
  -d '{"endpoint_id": "sensor-01", ...}'
```

## Contact
- GitHub: https://github.com/mossrake/ai-trust-chain
- Copyright: Mossrake Group, LLC (2025)
- License: AGPL-3.0 (implementation), Proprietary (framework design)
