#!/usr/bin/env python3
"""
AI Trust Chain REST API Client with AI Understanding
Script that creates assertions and uses Azure OpenAI to understand trust explanations
Usage: python trust_chain_client.py
"""

import requests
import json
import sys
import os
from datetime import datetime
from typing import List, Dict, Optional

# Configuration
BASE_URL = "http://localhost:5000"

# Azure OpenAI Configuration (set these as environment variables or update here)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-resource.openai.azure.com")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "your-api-key")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
AZURE_API_VERSION = "2024-02-15-preview"

def understand_trust_explanation(consumed_assertions: List[Dict], endpoint_type: str) -> Dict:
    """
    Use Azure OpenAI to understand trust explanations from consumed assertions
    This simulates what an endpoint would do to comprehend its inputs
    """
    if not AZURE_OPENAI_KEY or AZURE_OPENAI_KEY == "your-api-key":
        # Fallback if Azure OpenAI is not configured
        return {
            "understanding": "Azure OpenAI not configured - using default interpretation",
            "materiality_adjustments": {},
            "confidence_impact": 1.0,
            "trust_explanation": 'Operation nominal'
        }
    
    # Build context from consumed assertions
    context = f"As a {endpoint_type} endpoint, I need to understand these inputs:\n\n"
    for consumed_details in consumed_assertions:
        #print( f'Assertion: {consumed_details}')
        context += f"- Assertion {consumed_details['id']}:\n"
        context += f"  Trust: {consumed_details.get('trust_value')}\n"
        context += f"  Explanation: {consumed_details.get('trust_explanation')}\n"
        context += f"  Content: {json.dumps(consumed_details.get('content', {}))}\n\n"
    
    print( f'Context: {context}' )
    # Call Azure OpenAI to understand the implications
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_KEY
    }
    
    prompt = f"""{context}
    Based on these inputs, provide:
    1. A summary understanding of what these trust values and explanations mean for my operation
    2. How I should adjust materiality for each input (0.0-1.0 scale)
    3. How this should impact my own confidence (multiplication factor)
    4. A trust explanation describing the overall trust status
    
    Respond in JSON format with keys: understanding, materiality_adjustments, confidence_impact, trust_explanation"""
    
    try:
        response = requests.post(
            f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}",
            headers=headers,
            json={
                "messages": [
                    {"role": "system", "content": "You are an AI system helping endpoints understand trust implications."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 500
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            # Parse the LLM response
            llm_response = result['choices'][0]['message']['content']
            try:
                return json.loads(llm_response)
            except:
                return {
                    "understanding": llm_response,
                    "materiality_adjustments": {},
                    "confidence_impact": 0.9,
                    "trust_explanation": 'Processing with reduced confidence due to interpretation issues'
                }
        else:
            print(f"⚠️  Azure OpenAI call failed: {response.status_code}")
            return {
                "understanding": "Failed to get AI interpretation",
                "materiality_adjustments": {},
                "confidence_impact": 0.8,
                "trust_explanation": 'Processing with fallback logic due to AI unavailability'
             }
    except Exception as e:
        print(f"⚠️  Error calling Azure OpenAI: {e}")
        return {
            "understanding": "Error in AI interpretation",
            "materiality_adjustments": {},
            "confidence_impact": 0.8,
            "trust_explanation": 'Processing without AI understanding'
        }

def get_assertion_details(assertion_id, token="user-token") -> Optional[Dict]:
    """Get details of an assertion including trust explanation"""
    response = requests.get(
        f"{BASE_URL}/api/v1/assertions/{assertion_id}",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    if response.status_code == 200:
        return response.json()["data"]
    return None

def initialize_system(token="admin-token"):
    """Initialize the trust chain system"""
    print("Initializing trust chain system...")
    
    response = requests.post(
        f"{BASE_URL}/api/v1/admin/initialize",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        json={}
    )
    
    if response.status_code == 200:
        print("✅ System initialized successfully")
    else:
        print(f"⚠️  Initialize failed: {response.status_code} - {response.text}")

def set_trust_ceiling(endpoint_class, max_trust, token="admin-token"):
    """Set trust ceiling for an endpoint class"""
    print(f"\nSetting trust ceiling for {endpoint_class}...")
    
    response = requests.post(
        f"{BASE_URL}/api/v1/trust/endpoint-class",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        json={
            "endpoint_class": endpoint_class,
            "max_trust": max_trust
        }
    )
    
    if response.status_code == 200:
        print(f"✅ Trust ceiling set to {max_trust} for {endpoint_class}")
        return response.json()
    else:
        print(f"❌ Failed: {response.status_code} - {response.text}")
        return None

def create_door_sensor_assertion(door_id, is_open, confidence=0.99, token="sensor-token"):
    """Create a door sensor assertion (open/closed)"""
    print(f"\n🚪 Creating door sensor assertion ({door_id})...")
    
    response = requests.post(
        f"{BASE_URL}/api/v1/assertions",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        json={
            "endpoint_id": door_id,
            "endpoint_class": "sensor.door.magnetic_switch",
            "endpoint_type": "sensor",
            "content": {
                "status": "open" if is_open else "closed",
                "is_open": is_open,
                "timestamp": datetime.utcnow().isoformat()
            },
            "confidence": confidence,
            "trust_value": 1.0,  # Root sensor - provides its own trust
            "trust_explanation": "Magnetic switch sensor functioning normally"
        }
    )
    
    if response.status_code == 201:
        data = response.json()["data"]
        print(f"✅ Door sensor assertion created")
        print(f"   ID: {data['assertion_id']}")
        print(f"   Status: {'OPEN' if is_open else 'CLOSED'}")
        print(f"   Trust: {data.get('trust_value', 'N/A')}")
        print(f"   Confidence: {data.get('confidence_value', 'N/A')}")
        return data['assertion_id']
    else:
        print(f"❌ Failed: {response.status_code} - {response.text}")
        return None

def create_sensor_assertion(sensor_id, temperature, confidence=0.98, token="sensor-token"):
    """Create a sensor assertion"""
    print(f"\n📡 Creating sensor assertion ({sensor_id})...")
    
    print( f'{sensor_id}')
    response = requests.post(
        f"{BASE_URL}/api/v1/assertions",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        json={
            "endpoint_id": sensor_id,
            "endpoint_class": "sensor.temperature.honeywell_t7771a",
            "endpoint_type": "sensor",
            "content": {
                "temperature": temperature,
                "unit": "fahrenheit",
                "timestamp": datetime.utcnow().isoformat()
            },
            "confidence": confidence,
            "trust_value": 0.99,  # Root sensor - provides its own trust
            "trust_explanation": "Temperature sensor calibrated and functioning normally"
        }
    )
    
    if response.status_code == 201:
        data = response.json()["data"]
        print(f"✅ Sensor assertion created")
        print(f"   ID: {data['assertion_id']}")
        print(f"   Trust: {data.get('trust_value', 'N/A')}")
        print(f"   Confidence: {data.get('confidence_value', 'N/A')}")
        return data['assertion_id']
    else:
        print(f"❌ Failed: {response.status_code} - {response.text}")
        return None

def create_ml_assertion(sensor_ids, token="ml-token"):
    """Create an ML model assertion consuming multiple sensor data with AI understanding"""
    print("\n🤖 Creating ML model assertion with AI understanding...")
    
    # First, get details of consumed assertions to understand them
    consumed_details = []
    for sid in sensor_ids:
        details = get_assertion_details(sid)
        #print( f'{details}')
  
        if details:
            metadata = details['metadata']
            consumed_details.append({
                "id": sid,
                "trust_value": metadata.get("trust_value"),
                "trust_explanation": metadata.get("trust_explanation", "No explanation"),
                "content": details.get("content", {})
            })
    
    # Use AI to understand the trust implications
    #print( f'consumed datails {consumed_details}' )
    understanding = understand_trust_explanation(consumed_details, "ml_model")
    print(f"   AI Understanding: {understanding['understanding'][:100]}...")
    
    # Adjust materiality based on AI understanding
    materiality = {}
    for sid in sensor_ids:
        # Use AI-suggested materiality or default
        ai_materiality = understanding['materiality_adjustments'].get(sid, 0.9)
        materiality[sid] = ai_materiality
    
    # Adjust confidence based on AI understanding
    base_confidence = 0.85
    adjusted_confidence = base_confidence * understanding.get('confidence_impact', 1.0)
    
    # DO NOT send trust_value for ML assertions - server will calculate via propagation
    response = requests.post(
        f"{BASE_URL}/api/v1/assertions",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        json={
            "endpoint_id": "ml-model-01",
            "endpoint_class": "ml_model.predictive",
            "endpoint_type": "ml_model",
            "content": {
                "prediction": "maintenance_required",
                "probability": 0.82,
                "timestamp": datetime.utcnow().isoformat(),
                "analysis": "Based on readings from multiple temperature sensors",
            },
            "confidence": adjusted_confidence,
            "trust_value": 0,  # Send 0 to trigger propagation calculation
            "trust_explanation": understanding['trust_explanation'],
            "consumed_assertions": sensor_ids,
            "consumed_assertion_materiality": materiality
        }
    )
    
    if response.status_code == 201:
        data = response.json()["data"]
        print( f'ML data {data}')
        print(f"✅ ML assertion created")
        print(f"   ID: {data['assertion_id']}")
        print(f"   Trust: {data.get('trust_value', 'N/A')} (calculated by propagation)")
        print(f"   Confidence: {data.get('confidence_value', 'N/A')}")
        print(f"   Explanation: {data['trust_explanation']}")
        return data['assertion_id']
    else:
        print(f"❌ Failed: {response.status_code} - {response.text}")
        return None

def create_llm_assertion(ml_id, door_id, token="llm-token"):
    """Create an LLM assertion consuming ML predictions and door sensor with AI understanding"""
    print("\n🧠 Creating LLM assertion with AI understanding...")
    
    # Get details of consumed assertions
    consumed_details = []
    
    ml_details = get_assertion_details(ml_id)
    if ml_details:
        metadata = ml_details.get('metadata', {})  # Get metadata field

        consumed_details.append({
            "id": ml_id,
            "trust_value": metadata.get("trust_value"),
            "trust_explanation": metadata.get("trust_explanation", "No explanation"),
            "content": ml_details.get("content", {})
        })
    
    door_details = get_assertion_details(door_id)
    if door_details:
        metadata = door_details.get('metadata', {})  # Get metadata field
        
        consumed_details.append({
            "id": door_id,
            "trust_value": metadata.get("trust_value"),
            "trust_explanation": metadata.get("trust_explanation", "No explanation"),
            "content": door_details.get("content", {})
        })
    
    # Use AI to understand and synthesize all inputs
    understanding = understand_trust_explanation(consumed_details, "llm")
    print(f"   AI Synthesis: {understanding['understanding']}...")
    
    # Build intelligent materiality based on understanding
    materiality = {
        ml_id: understanding['materiality_adjustments'].get(ml_id, 0.95),
        door_id: understanding['materiality_adjustments'].get(door_id, 0.80)
    }
    
    # Generate intelligent recommendation based on trust understanding
    if understanding.get('confidence_impact', 1.0) < 0.7:
        recommendation = "Requires immediate human review due to trust concerns"
        risk_level = "high"
    else:
        recommendation = "Schedule maintenance within 48 hours; ensure refrigerator door is properly sealed"
        risk_level = "medium"

    print( f'***recommendation***\n{recommendation}\n' )
    print( f'trying the server ...')
     
    # DO NOT send trust_value for LLM assertions - server will calculate via propagation
    response = requests.post(
        f"{BASE_URL}/api/v1/assertions",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        json={
            "endpoint_id": "gpt4-analyzer",
            "endpoint_class": "llm.gpt4",
            "endpoint_type": "llm",
            "content": {
                "recommendation": recommendation,
                "risk_level": risk_level,
                "analysis": "Based on temperature sensors, ML predictions, and door status",
                "timestamp": datetime.utcnow().isoformat()
            },
            "confidence": 0.88 * understanding.get('confidence_impact', 1.0),
            "trust_value": 0,  # Send 0 to trigger propagation calculation
            "trust_explanation": understanding['trust_explanation'],
            "consumed_assertions": [ml_id, door_id],
            "consumed_assertion_materiality": materiality
        }
    )
    print( f'status code: {response.status_code}')
    if response.status_code == 201:
        data = response.json()["data"]
        print(f"✅ LLM assertion created")
        print(f"   ID: {data['assertion_id']}")
        print(f"   Trust: {data.get('trust_value', 'N/A')} (calculated by propagation)")
        print(f"   Confidence: {data.get('confidence_value', 'N/A')}")
        print(f"   Explanation: {data['trust_explanation']}")
        return data['assertion_id']
    else:
        print(f"❌ Failed: {response.status_code} - {response.text}")
        return None

def get_evidence(assertion_id, token="user-token"):
    """Get and display the evidence tree for an assertion"""
    print(f"\n🔍 Fetching evidence for assertion: {assertion_id}")
    
    response = requests.get(
        f"{BASE_URL}/api/v1/assertions/{assertion_id}/evidence",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    if response.status_code == 200:
        data = response.json()["data"]
        
        print("\n" + "="*70)
        print("EVIDENCE CHAIN SUMMARY")
        print("="*70)
        
        summary = data["summary"]
        print(f"📊 Chain Depth: {summary['chain_depth']}")
        print(f"📊 Total Assertions: {summary['total_assertions']}")
        
        weakest = summary["weakest_link"]
        print(f"\n⚠️  Weakest Link:")
        print(f"   - Endpoint: {weakest['endpoint_id']}")
        print(f"   - Trust Value: {weakest['trust_value']}")
        print(f"   - Explanation: {weakest['explanation']}")
        
        print(f"\n💡 Recommendation: {summary['recommendation']}")
        
        print("\n" + "="*70)
        print("EVIDENCE TREE")
        print("="*70)
        print(json.dumps(data["evidence_tree"], indent=2))
        
        return data
    else:
        print(f"❌ Failed to get evidence: {response.status_code} - {response.text}")
        return None

def get_explanation(assertion_id, token="user-token"):
    """Get human-readable explanation of evidence chain"""
    print(f"\n📝 Getting explanation for assertion: {assertion_id}")
    
    response = requests.get(
        f"{BASE_URL}/api/v1/assertions/{assertion_id}/explain",
        headers={"Authorization": f"Bearer {token}"},
        params={"format": "markdown"}
    )
    
    if response.status_code == 200:
        data = response.json()["data"]
        print("\n" + "="*70)
        print("HUMAN-READABLE EXPLANATION")
        print("="*70)
        print(data["explanation"])
        return data
    else:
        print(f"❌ Failed to get explanation: {response.status_code} - {response.text}")
        return None

def get_trust_matrix(assertion_id, token="user-token"):
    """Get trust-confidence matrix classification"""
    print(f"\n📊 Getting trust matrix for assertion: {assertion_id}")
    
    response = requests.get(
        f"{BASE_URL}/api/v1/assertions/{assertion_id}/matrix",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    if response.status_code == 200:
        data = response.json()["data"]
        print("\n" + "="*70)
        print("TRUST-CONFIDENCE MATRIX")
        print("="*70)
        print(f"Status: {data['status']}")
        print(f"Recommendation: {data['recommendation']}")
        print(f"Trust Value: {data['trust_value']}")
        print(f"Confidence Value: {data['confidence_value']}")
        
        # Decision logic based on matrix
        if data['status'] == 'high_trust_high_confidence':
            print("\n✅ Safe to proceed automatically")
        elif data['status'] == 'high_trust_low_confidence':
            print("\n⚠️  Human review recommended for ambiguous situation")
        elif data['status'] == 'low_trust_high_confidence':
            print("\n🛑 Exercise extreme caution - potential overconfidence")
        else:  # low_trust_low_confidence
            print("\n❌ Seek alternative information sources")
        
        return data
    else:
        print(f"❌ Failed to get matrix: {response.status_code} - {response.text}")
        return None

def main():
    """Main execution flow"""
    print("="*70)
    print("AI TRUST CHAIN DEMONSTRATION WITH AI UNDERSTANDING")
    print("="*70)
    
    # Check Azure OpenAI configuration
    if AZURE_OPENAI_KEY == "your-api-key":
        print("\n⚠️  WARNING: Azure OpenAI not configured!")
        print("Set environment variables:")
        print("  export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com'")
        print("  export AZURE_OPENAI_KEY='your-api-key'")
        print("  export AZURE_OPENAI_DEPLOYMENT='your-deployment-name'")
        print("\nContinuing with fallback mode...\n")
    
    # Step 1: Initialize system (optional, may already be initialized)
    print("\n1️⃣  SYSTEM INITIALIZATION")
    print("-"*40)
    # Uncomment if you need to initialize
    # initialize_system()
    
    # Step 2: Set trust ceilings (optional)
    print("\n2️⃣  SETTING TRUST CEILINGS")
    print("-"*40)
    set_trust_ceiling("sensor.temperature.honeywell_t7771a", 0.90)
    set_trust_ceiling("ml_model.predictive", 0.85)
    set_trust_ceiling("llm.gpt4", 0.80)
    
    # Step 3: Create assertion chain
    print("\n3️⃣  CREATING ASSERTION CHAIN")
    print("-"*40)
    
    # Create first temperature sensor assertion
    sensor1_id = create_sensor_assertion("temp-sensor-01", 75.2, 0.98)
    if not sensor1_id:
        print("Failed to create sensor 1 assertion. Exiting.")
        sys.exit(1)
    
    # Create second temperature sensor assertion
    sensor2_id = create_sensor_assertion("temp-sensor-02", 76.8, 0.95)
    if not sensor2_id:
        print("Failed to create sensor 2 assertion. Exiting.")
        sys.exit(1)
    
    # Create door sensor assertion (refrigerator door)
    door_id = create_door_sensor_assertion("fridge-door-01", is_open=True, confidence=0.99)
    if not door_id:
        print("Failed to create door sensor assertion. Exiting.")
        sys.exit(1)
    
    # Create ML model assertion consuming both temperature sensors
    ml_id = create_ml_assertion([sensor1_id, sensor2_id])
    if not ml_id:
        print("Failed to create ML assertion. Exiting.")
        sys.exit(1)
    
    # Create LLM assertion consuming ML prediction AND door sensor
    llm_id = create_llm_assertion(ml_id, door_id)
    if not llm_id:
        print("Failed to create LLM assertion. Exiting.")
        sys.exit(1)
    
    # Step 4: Display evidence chain
    print("\n4️⃣  DISPLAYING EVIDENCE")
    print("-"*40)
    get_evidence(llm_id)
    
    # Step 5: Get human-readable explanation
    print("\n5️⃣  GETTING EXPLANATION")
    print("-"*40)
    get_explanation(llm_id)
    
    # Step 6: Get trust matrix
    print("\n6️⃣  TRUST-CONFIDENCE MATRIX ANALYSIS")
    print("-"*40)
    get_trust_matrix(llm_id)
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print(f"\nFinal assertion ID: {llm_id}")
    print("You can use this ID to query the API for more information.")

if __name__ == "__main__":
    main()
