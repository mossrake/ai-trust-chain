# AI Trust Chain Framework - API Documentation

## Quick Start

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

## HTTP Status Codes

- `200 OK` - Successful GET request
- `201 Created` - Successful resource creation
- `400 Bad Request` - Invalid request parameters
- `401 Unauthorized` - Missing or invalid authentication
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error

## Trust-Confidence Matrix

The framework provides decision guidance based on trust-confidence combinations:

| Trust | Confidence | Status | Recommendation |
|-------|------------|--------|----------------|
| High (≥0.7) | High (≥0.7) | Green | Proceed with minimal oversight |
| High (≥0.7) | Low (<0.7) | Yellow | Apply human judgment to ambiguous situations |
| Low (<0.7) | High (≥0.7) | Red | Exercise extreme caution due to potential overconfidence |
| Low (<0.7) | Low (<0.7) | Gray | Seek alternative information sources or defer decisions |
