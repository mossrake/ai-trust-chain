# AI Trust Chain Framework

An implementation of the AI Trust Chain framework for auditable decision support systems.

## Overview

The AI Trust Chain framework addresses the fundamental challenge of making confident, defensible decisions based on AI recommendations while maintaining transparency about supporting evidence. By separating confidence from trust, the framework protects against recommendations that appear certain but rest on untrustworthy foundations.

## Core Concepts

### Trust vs Confidence

- **Confidence**: An endpoint's self-reported certainty in its assertion
- **Trust**: The system's assessment of an endpoint's assertions based on historical performance, contextual appropriateness, and known limitations

### Trust Propagation

Trust flows through the system following precise rules:
1. Trust values of consumed assertions
2. Materiality weights for each input
3. Endpoint-specific trust ceilings
4. Propagation factors

### Immutable Audit Trail

Every assertion is recorded in a blockchain-based ledger, providing:
- Complete traceability from recommendations to source data
- Cryptographic proof of unaltered trust evaluations
- Comprehensive accountability for AI-informed decisions

## Architecture Decisions

### Endpoint-Managed Trust Understanding

The AI Trust Chain framework delegates trust explanation interpretation to individual endpoints rather than handling it in the kernel. This design decision provides several benefits:

1. **Separation of Concerns**: The kernel focuses solely on trust calculation and propagation mechanics, while endpoints handle domain-specific interpretation.

2. **Flexibility**: Each endpoint can use its own AI/LLM service (Azure OpenAI, AWS Bedrock, local models, etc.) to understand trust implications relevant to its domain.

3. **Context-Aware Materiality**: Endpoints can dynamically adjust materiality based on their understanding of trust explanations from consumed assertions.

4. **Domain Expertise**: Endpoints best understand how trust issues in their inputs affect their outputs.

#### Implementation Pattern

Endpoints that need to understand trust explanations should:

1. Retrieve trust explanations from consumed assertions
2. Call their preferred AI service to interpret implications
3. Adjust materiality and confidence based on understanding
4. Include synthesized understanding in their own assertion

Example:
```python
# In endpoint implementation
def create_assertion_with_understanding(consumed_assertions):
    # Get trust details from consumed assertions
    trust_details = [get_assertion_details(id) for id in consumed_assertions]
    
    # Use AI to understand implications
    understanding = call_ai_service(trust_details, endpoint_context)
    
    # Adjust materiality based on understanding
    materiality = calculate_materiality(understanding)
    
    # Create assertion with informed decisions
    return create_assertion(
        content=process_data(),
        materiality=materiality,
        confidence=adjusted_confidence(understanding)
    )
```

This approach keeps the kernel lightweight while enabling sophisticated trust comprehension at the endpoint level.

## Quick Start

### Prerequisites

- Python 3.8+
- PostgreSQL 12+ (or SQLite for development)
- Redis (optional, for caching)

### Installation

```bash
# Clone the repository
git clone https://github.com/mossrake/ai-trust-chain.git
cd ai-trust-chain

# Install dependencies
pip install -r requirements.txt
```

### 1. Start the REST API Server

```bash
python trust_rest_api.py
```

The server will start on `http://localhost:5000`

### 2. Initialize the System

```bash
curl -X POST http://localhost:5000/api/v1/admin/initialize \
  -H "Authorization: Bearer admin-token" \
  -H "Content-Type: application/json" \
  -d '{}'
```

### 3. Create an Assertion

```bash
curl -X POST http://localhost:5000/api/v1/assertions \
  -H "Authorization: Bearer user-token" \
  -H "Content-Type: application/json" \
  -d '{
    "endpoint_id": "sensor-001",
    "endpoint_class": "sensor.temperature.honeywell_t7771a",
    "endpoint_type": "sensor",
    "content": {"temperature": 72.5, "unit": "fahrenheit"},
    "confidence": 0.95
  }'
```

### 4. View the evidence chain
```bash
curl -X GET http://localhost:5000/api/v1/assertions/{id}/evidence \
  -H "Authorization: Bearer user-token"

## API Documentation

### Authentication

The API uses **pass-through authentication by design** - credentials are validated on each request with no local storage. This stateless approach:
- Integrates with existing enterprise auth systems
- Requires no credential management
- Maintains zero auth state in the trust chain
- Enables seamless SSO/OAuth integration

Include an `Authorization` header with each request:
```
Authorization: Bearer <your-token>
```

Admin endpoints require tokens that identify admin privileges (implementation-specific).

### Core Endpoints

#### Trust Registry Management
- `GET /api/v1/trust/registry` - Get current trust configuration
- `POST /api/v1/trust/endpoint-class` - Set trust ceiling for endpoint class (admin)
- `POST /api/v1/trust/propagation-rule` - Add/update propagation rule (admin)
- `DELETE /api/v1/trust/propagation-rule/<rule_id>` - Delete propagation rule (admin)

#### Assertion Management
- `POST /api/v1/assertions` - Create new assertion with trust evaluation
- `GET /api/v1/assertions/<id>` - Retrieve assertion by ID
- `GET /api/v1/assertions/<id>/provenance` - Get complete provenance chain (legacy)
- `GET /api/v1/assertions/<id>/evidence` - **"Show me the evidence"** - Complete evidence tree
- `GET /api/v1/assertions/<id>/explain` - Human-readable explanation of evidence chain
- `GET /api/v1/assertions/<id>/matrix` - Get trust-confidence matrix classification

#### Blockchain Operations
- `POST /api/v1/blockchain/commit` - Manually commit pending assertions
- `GET /api/v1/blockchain/verify` - Verify blockchain integrity
- `GET /api/v1/blockchain/latest` - Get latest block information

#### Administration
- `POST /api/v1/admin/initialize` - Initialize with default configuration (admin)
- `POST /api/v1/admin/reset` - Reset trust registry (development only, admin)

## Usage Examples

### Shell/cURL Examples

#### Setting Trust Ceiling for an Endpoint Class

```bash
curl -X POST http://localhost:5000/api/v1/trust/endpoint-class \
  -H "Authorization: Bearer admin-token" \
  -H "Content-Type: application/json" \
  -d '{
    "endpoint_class": "llm.gpt5",
    "max_trust": 0.85
  }'
```

#### Creating a Chain of Assertions

First, create a sensor assertion:

```bash
curl -X POST http://localhost:5000/api/v1/assertions \
  -H "Authorization: Bearer sensor-token" \
  -H "Content-Type: application/json" \
  -d '{
    "endpoint_id": "temp-sensor-01",
    "endpoint_class": "sensor.temperature.honeywell_t7771a",
    "endpoint_type": "sensor",
    "content": {"temperature": 75.2},
    "confidence": 0.98
  }'
```

Save the returned `assertion_id` from the response, then create an ML model assertion that consumes it:

```bash
curl -X POST http://localhost:5000/api/v1/assertions \
  -H "Authorization: Bearer ml-token" \
  -H "Content-Type: application/json" \
  -d '{
    "endpoint_id": "ml-model-01",
    "endpoint_class": "ml_model.predictive",
    "endpoint_type": "ml_model",
    "content": {"prediction": "maintenance_required", "probability": 0.82},
    "confidence": 0.85,
    "consumed_assertions": ["SENSOR_ASSERTION_ID_HERE"],
    "consumed_assertion_materiality": {"SENSOR_ASSERTION_ID_HERE": 0.9}
  }'
```

#### Getting Evidence for an Assertion

```bash
curl -X GET http://localhost:5000/api/v1/assertions/ASSERTION_ID_HERE/evidence \
  -H "Authorization: Bearer user-token"
```

#### Getting Human-Readable Explanation

```bash
curl -X GET "http://localhost:5000/api/v1/assertions/ASSERTION_ID_HERE/explain?format=markdown" \
  -H "Authorization: Bearer user-token"
```

#### Getting Trust-Confidence Matrix Decision

```bash
curl -X GET http://localhost:5000/api/v1/assertions/ASSERTION_ID_HERE/matrix \
  -H "Authorization: Bearer user-token"
```

### Python Examples

#### Setting Trust Ceiling for an Endpoint Class

```python
import requests

# Configure the API endpoint and headers
base_url = "http://localhost:5000"
headers = {
    "Authorization": "Bearer admin-token",
    "Content-Type": "application/json"
}

# Set trust ceiling for an endpoint class
data = {
    "endpoint_class": "llm.gpt5",
    "max_trust": 0.85
}

response = requests.post(
    f"{base_url}/api/v1/trust/endpoint-class",
    json=data,
    headers=headers
)

if response.status_code == 200:
    print("Trust ceiling set successfully")
    print(response.json())
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

#### Creating a Chain of Assertions

```python
import requests

base_url = "http://localhost:5000"

# Step 1: Create sensor assertion
sensor_headers = {
    "Authorization": "Bearer sensor-token",
    "Content-Type": "application/json"
}

sensor_data = {
    "endpoint_id": "temp-sensor-01",
    "endpoint_class": "sensor.temperature.honeywell_t7771a",
    "endpoint_type": "sensor",
    "content": {"temperature": 75.2},
    "confidence": 0.98
}

sensor_response = requests.post(
    f"{base_url}/api/v1/assertions",
    json=sensor_data,
    headers=sensor_headers
)

if sensor_response.status_code != 201:
    print(f"Error creating sensor assertion: {sensor_response.text}")
    exit(1)

sensor_id = sensor_response.json()["data"]["assertion_id"]
print(f"Created sensor assertion: {sensor_id}")

# Step 2: Create ML model assertion consuming sensor data
ml_headers = {
    "Authorization": "Bearer ml-token",
    "Content-Type": "application/json"
}

ml_data = {
    "endpoint_id": "ml-model-01",
    "endpoint_class": "ml_model.predictive",
    "endpoint_type": "ml_model",
    "content": {
        "prediction": "maintenance_required",
        "probability": 0.82
    },
    "confidence": 0.85,
    "consumed_assertions": [sensor_id],
    "consumed_assertion_materiality": {
        sensor_id: 0.9  # High materiality
    }
}

ml_response = requests.post(
    f"{base_url}/api/v1/assertions",
    json=ml_data,
    headers=ml_headers
)

if ml_response.status_code == 201:
    ml_result = ml_response.json()["data"]
    print(f"Created ML assertion: {ml_result['assertion_id']}")
    if "trust_explanation" in ml_result:
        print(f"Trust explanation: {ml_result['trust_explanation']}")
else:
    print(f"Error creating ML assertion: {ml_response.text}")
```

#### "Show Me The Evidence" - Tracing Decision Chains

```python
import requests
import json

base_url = "http://localhost:5000"
assertion_id = "your-assertion-id-here"
headers = {
    "Authorization": "Bearer user-token"
}

# Get the complete evidence tree
response = requests.get(
    f"{base_url}/api/v1/assertions/{assertion_id}/evidence",
    headers=headers
)

if response.status_code == 200:
    evidence = response.json()["data"]
    
    # Display summary information
    summary = evidence["summary"]
    print(f"Chain depth: {summary['chain_depth']}")
    print(f"Total assertions: {summary['total_assertions']}")
    
    # Show weakest link in the chain
    weakest = summary["weakest_link"]
    print(f"\nWeakest link:")
    print(f"  Endpoint: {weakest['endpoint_id']}")
    print(f"  Trust: {weakest['trust_value']}")
    print(f"  Issue: {weakest['explanation']}")
    
    # Display recommendation
    print(f"\nRecommendation: {summary['recommendation']}")
    
    # Pretty print the full evidence tree
    print("\nFull Evidence Tree:")
    print(json.dumps(evidence["evidence_tree"], indent=2))
else:
    print(f"Error: {response.status_code}")
    print(response.text)

# Get human-readable explanation
response = requests.get(
    f"{base_url}/api/v1/assertions/{assertion_id}/explain",
    params={"format": "markdown"},
    headers=headers
)

if response.status_code == 200:
    explanation = response.json()["data"]["explanation"]
    print("\nHuman-Readable Explanation:")
    print(explanation)
```

#### Getting Trust-Confidence Matrix Classification

```python
import requests

base_url = "http://localhost:5000"
assertion_id = "your-assertion-id-here"
headers = {
    "Authorization": "Bearer user-token"
}

response = requests.get(
    f"{base_url}/api/v1/assertions/{assertion_id}/matrix",
    headers=headers
)

if response.status_code == 200:
    matrix_data = response.json()["data"]
    print(f"Status: {matrix_data['status']}")
    print(f"Recommendation: {matrix_data['recommendation']}")
    print(f"Trust: {matrix_data['trust_value']}")
    print(f"Confidence: {matrix_data['confidence_value']}")
    
    # Decision logic based on matrix
    if matrix_data['status'] == 'high_trust_high_confidence':
        print("✅ Safe to proceed automatically")
    elif matrix_data['status'] == 'high_trust_low_confidence':
        print("⚠️ Human review recommended for ambiguous situation")
    elif matrix_data['status'] == 'low_trust_high_confidence':
        print("🛑 Exercise extreme caution - potential overconfidence")
    else:  # low_trust_low_confidence
        print("❌ Seek alternative information sources")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

## Response Examples

### Successful Assertion Creation Response

```json
{
  "status": "success",
  "data": {
    "assertion_id": "abc-123-def-456",
    "trust_value": 0.75,
    "confidence_value": 0.90,
    "temporal_validity": 0.95,
    "trust_explanation": "Trust is moderate (0.75) due to temperature sensor calibration being 6 months old, though recent readings show good consistency",
    "timestamp": "2025-01-15T10:30:00Z"
  }
}
```

### Evidence Tree Response

```json
{
  "status": "success",
  "data": {
    "evidence_tree": {
      "assertion_id": "llm-rec-001",
      "endpoint_id": "gpt4-analyzer",
      "endpoint_class": "llm.gpt4",
      "trust_value": 0.68,
      "confidence_value": 0.85,
      "trust_explanation": "Trust reduced due to stale temperature data",
      "content": {
        "recommendation": "Schedule maintenance within 48 hours",
        "risk_level": "medium"
      },
      "consumed_assertions": [
        {
          "assertion_id": "ml-pred-001",
          "endpoint_id": "failure-predictor",
          "endpoint_class": "ml_model.predictive",
          "trust_value": 0.70,
          "confidence_value": 0.82,
          "content": {
            "prediction": "maintenance_required",
            "probability": 0.82
          },
          "consumed_assertions": [
            {
              "assertion_id": "sensor-001",
              "endpoint_id": "temp-sensor-01",
              "endpoint_class": "sensor.temperature.honeywell_t7771a",
              "trust_value": 0.70,
              "confidence_value": 0.95,
              "trust_explanation": "Sensor uncalibrated for 6 months",
              "content": {
                "temperature": 75.2
              },
              "consumed_assertions": []
            }
          ]
        }
      ]
    },
    "summary": {
      "chain_depth": 3,
      "total_assertions": 3,
      "weakest_link": {
        "endpoint_id": "temp-sensor-01",
        "trust_value": 0.70,
        "explanation": "Sensor uncalibrated for 6 months"
      },
      "recommendation": "Apply human judgment to ambiguous situations"
    }
  }
}
```

### Trust-Confidence Matrix Response

```json
{
  "status": "success",
  "data": {
    "status": "high_trust_high_confidence",
    "recommendation": "Proceed with minimal oversight",
    "trust_value": 0.75,
    "confidence_value": 0.90,
    "thresholds": {
      "trust_threshold": 0.7,
      "confidence_threshold": 0.7
    }
  }
}
```

### Error Response

```json
{
  "status": "error",
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Invalid or missing authorization token"
  }
}
```


## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Endpoints                       │
│  (Sensors, APIs, ML Models, LLMs)               │
│  - Create assertions                             │
│  - Understand trust explanations                 │
│  - Apply materiality                             │
└─────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│              Trust Kernel                        │
│  - Calculate trust propagation                   │
│  - Apply trust ceilings                          │
│  - Generate basic explanations                   │
└─────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│           Blockchain Ledger                      │
│  - Immutable assertion storage                   │
│  - Cryptographic verification                    │
│  - Complete audit trail                          │
└─────────────────────────────────────────────────┘
```

## Contributing

At this time we are not accepting contributions.  In the future, please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{ai_trust_chain,
  title = {AI Trust Chain: A Framework for Auditable Decision Support},
  author = {Mossrake Group, LLC},
  year = {2025},
  url = {https://github.com/mossrake/ai-trust-chain}
}
```

## Support

- Documentation: [https://mossrake.com/ai-trust-chain](https://mossrake.com/ai-trust-chain)
- Issues: [GitHub Issues](https://github.com/mossrake/ai-trust-chain/issues)
- ##Discussion: [GitHub Discussions](https://github.com/mossrake/ai-trust-chain/discussions)

## Acknowledgments

- Based on the AI Trust Chain framework whitepaper
- Inspired by challenges in enterprise AI adoption
- Built for transparency and accountability in AI systems