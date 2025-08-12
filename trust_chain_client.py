#!/usr/bin/env python3
"""
AI Trust Chain REST API Client with Enhanced Reasoning Output
This version provides detailed explanations of the trust propagation logic

Copyright (C) 2025 Mossrake Group, LLC
Released under AGPL-3.0
"""

import requests
import json
import sys
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import time

# Configuration
BASE_URL = "http://localhost:5000"

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip('/')
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_reasoning(title: str, content: str, color=Colors.CYAN):
    """Print reasoning with formatting"""
    print(f"\n{color}🧠 {title}{Colors.ENDC}")
    print(f"{Colors.BOLD}   Reasoning:{Colors.ENDC} {content}")

def print_trust_flow(from_trust: float, to_trust: float, reason: str):
    """Show how trust changed"""
    symbol = "↓" if to_trust < from_trust else "↑" if to_trust > from_trust else "→"
    color = Colors.FAIL if to_trust < from_trust else Colors.GREEN if to_trust > from_trust else Colors.BLUE
    print(f"\n{color}   Trust Flow: {from_trust:.3f} {symbol} {to_trust:.3f}{Colors.ENDC}")
    print(f"   {Colors.BOLD}Why:{Colors.ENDC} {reason}")

def understand_trust_explanation(consumed_assertions: List[Dict], endpoint_type: str) -> Dict:
    """
    Use Azure OpenAI to understand trust explanations from consumed assertions
    and determine appropriate materiality specifications
    """
    print(f"\n{Colors.CYAN}🤖 AI REASONING PROCESS{Colors.ENDC}")
    print(f"   Analyzing {len(consumed_assertions)} inputs for {endpoint_type} endpoint...")
    
    if not AZURE_OPENAI_KEY or AZURE_OPENAI_KEY == "your-api-key":
        print(f"   {Colors.WARNING}⚠️  Azure OpenAI not configured - using fallback logic{Colors.ENDC}")
        
        # Explain fallback reasoning
        default_specs = {}
        num_assertions = len(consumed_assertions) if consumed_assertions else 1
        
        print(f"\n   {Colors.BOLD}Fallback Reasoning:{Colors.ENDC}")
        for consumed in consumed_assertions:
            is_debug = 'debug' in consumed['id'].lower() or 'api.debug' in str(consumed.get('content', ''))
            if is_debug:
                print(f"   • {consumed['id']}: Identified as debug data → context only (0.0, False)")
                default_specs[consumed['id']] = (0.0, False)
            else:
                weight = 1.0 / len([c for c in consumed_assertions if 'debug' not in c['id'].lower()])
                print(f"   • {consumed['id']}: Regular input → affects trust (weight={weight:.2f}, True)")
                default_specs[consumed['id']] = (weight, True)
        
        return {
            "understanding": "Using rule-based reasoning without AI enhancement",
            "materiality_specs": default_specs,
            "confidence_impact": 0.9,
            "trust_explanation": 'Trust propagated using equal weighting for non-debug inputs'
        }
    
    # Build context for AI
    print(f"\n   {Colors.BOLD}Preparing context for AI analysis...{Colors.ENDC}")
    context = f"As a {endpoint_type} endpoint, I need to understand these inputs:\n\n"
    
    for consumed_details in consumed_assertions:
        trust_val = consumed_details.get('trust_value', 'N/A')
        print(f"   • Input: {consumed_details['id']} (Trust: {trust_val})")
        
        context += f"- Assertion {consumed_details['id']}:\n"
        context += f"  Trust: {trust_val}\n"
        context += f"  Explanation: {consumed_details.get('trust_explanation')}\n"
        context += f"  Content: {json.dumps(consumed_details.get('content', {}))}\n\n"
    
    print(f"\n   {Colors.BOLD}Calling Azure OpenAI for intelligent analysis...{Colors.ENDC}")
    
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_KEY
    }
    
    prompt = f"""{context}
    Based on these inputs, provide a JSON response analyzing the trust implications.
    
    Your response must be valid JSON with this exact structure:
    {{
        "understanding": "A summary of what these trust values mean",
        "materiality_specs": {{
            "assertion_id_here": {{
                "weight": 0.5,
                "apply_to_trust": true,
                "reason": "why this weight was chosen"
            }}
        }},
        "confidence_impact": 0.9,
        "trust_explanation": "Overall trust status explanation"
    }}
    
    Guidelines:
    - weight: 0.0-1.0 indicating importance
    - apply_to_trust: true if it should affect trust, false for context only
    - confidence_impact: multiplication factor for confidence
    - All weights for trust-affecting inputs should sum to approximately 1.0
    
    Important: Return ONLY valid JSON, no additional text or explanation outside the JSON structure."""
    
    try:
        response = requests.post(
            f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}",
            headers=headers,
            json={
                "messages": [
                    {"role": "system", "content": "You are an AI system helping endpoints understand trust implications and determine materiality."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 500
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            llm_response = result['choices'][0]['message']['content']
            
            try:
                ai_result = json.loads(llm_response)
                
                print(f"\n   {Colors.GREEN}✅ AI Analysis Complete:{Colors.ENDC}")
                print(f"   {Colors.BOLD}Understanding:{Colors.ENDC} {ai_result.get('understanding', '')[:150]}...")
                
                # Convert and explain AI's materiality decisions
                tuple_specs = {}
                print(f"\n   {Colors.BOLD}AI Materiality Decisions:{Colors.ENDC}")
                
                for aid, spec in ai_result.get('materiality_specs', {}).items():
                    weight = spec.get('weight', 0.5)
                    apply = spec.get('apply_to_trust', True)
                    reason = spec.get('reason', 'No reason provided')
                    
                    tuple_specs[aid] = (weight, apply)
                    
                    impact = "AFFECTS TRUST" if apply else "CONTEXT ONLY"
                    color = Colors.GREEN if apply else Colors.WARNING
                    print(f"   • {aid}: {color}{impact}{Colors.ENDC}")
                    print(f"     Weight: {weight:.2%}, Reason: {reason}")
                
                confidence_impact = ai_result.get('confidence_impact', 0.9)
                print(f"\n   {Colors.BOLD}Confidence Impact:{Colors.ENDC} {confidence_impact:.2f}x multiplier")
                print(f"   {Colors.BOLD}Trust Explanation:{Colors.ENDC} {ai_result.get('trust_explanation', '')}")
                
                return {
                    "understanding": ai_result.get('understanding', ''),
                    "materiality_specs": tuple_specs,
                    "confidence_impact": confidence_impact,
                    "trust_explanation": ai_result.get('trust_explanation', 'AI-assisted trust evaluation')
                }
                
            except Exception as e:
                print(f"   {Colors.WARNING}⚠️  Could not parse AI response, using fallback{Colors.ENDC}")
                print(f"   Error: {str(e)}")
                
                # Fallback with explanation
                default_specs = {}
                for consumed in consumed_assertions:
                    default_specs[consumed['id']] = (1.0 / len(consumed_assertions), True)
                
                return {
                    "understanding": llm_response[:200] if llm_response else "Parse error",
                    "materiality_specs": default_specs,
                    "confidence_impact": 0.9,
                    "trust_explanation": 'Processing with reduced confidence due to interpretation issues'
                }
        else:
            print(f"   {Colors.FAIL}❌ Azure OpenAI call failed: {response.status_code}{Colors.ENDC}")
            # Fallback
            default_specs = {}
            for consumed in consumed_assertions:
                default_specs[consumed['id']] = (0.8, True)
            
            return {
                "understanding": "Failed to get AI interpretation",
                "materiality_specs": default_specs,
                "confidence_impact": 0.8,
                "trust_explanation": 'Processing with fallback logic due to AI unavailability'
            }
            
    except Exception as e:
        print(f"   {Colors.FAIL}❌ Error calling Azure OpenAI: {e}{Colors.ENDC}")
        default_specs = {}
        for consumed in consumed_assertions:
            default_specs[consumed['id']] = (0.7, True)
        
        return {
            "understanding": "Error in AI interpretation",
            "materiality_specs": default_specs,
            "confidence_impact": 0.8,
            "trust_explanation": 'Processing without AI enhancement'
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
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}INITIALIZING TRUST CHAIN SYSTEM{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")
    
    print_reasoning(
        "System Initialization",
        "Setting up the trust authority and blockchain audit trail.\n" +
        "   This creates the foundation for immutable trust tracking."
    )
    
    response = requests.post(
        f"{BASE_URL}/api/v1/admin/initialize",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        json={}
    )
    
    if response.status_code == 200:
        print(f"{Colors.GREEN}✅ System initialized successfully{Colors.ENDC}")
    else:
        print(f"{Colors.FAIL}⚠️  Initialize failed: {response.status_code} - {response.text}{Colors.ENDC}")

def set_trust_ceiling(endpoint_class, max_trust, token="admin-token"):
    """Set trust ceiling for an endpoint class"""
    print(f"\n{Colors.BLUE}📏 Setting Trust Ceiling{Colors.ENDC}")
    print(f"   Endpoint Class: {endpoint_class}")
    print(f"   Maximum Trust: {max_trust}")
    
    print_reasoning(
        "Trust Ceiling Logic",
        f"This endpoint class can never exceed {max_trust} trust, even if all\n" +
        f"   inputs are perfect. This reflects inherent limitations of {endpoint_class.split('.')[0]}s."
    )
    
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
        print(f"{Colors.GREEN}✅ Trust ceiling set successfully{Colors.ENDC}")
        return response.json()
    else:
        print(f"{Colors.FAIL}❌ Failed: {response.status_code} - {response.text}{Colors.ENDC}")
        return None

def create_door_sensor_assertion(door_id, is_open, confidence=0.99, token="sensor-token"):
    """Create a door sensor assertion (open/closed) - ROOT assertion"""
    print(f"\n{Colors.CYAN}🚪 Creating Door Sensor Assertion{Colors.ENDC}")
    print(f"   Sensor ID: {door_id}")
    print(f"   Status: {'OPEN' if is_open else 'CLOSED'}")
    print(f"   Confidence: {confidence}")
    
    print_reasoning(
        "Root Assertion",
        "This is a ROOT sensor - it has no dependencies.\n" +
        "   Trust will be set to the endpoint class ceiling (not self-reported).\n" +
        "   This prevents sensors from claiming unwarranted high trust."
    )
    
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
            "trust_explanation": "Magnetic switch sensor functioning normally"
        }
    )
    
    if response.status_code == 201:
        data = response.json()["data"]
        print(f"{Colors.GREEN}✅ Door sensor assertion created{Colors.ENDC}")
        print(f"   ID: {data['assertion_id']}")
        print(f"   Trust: {data.get('trust_value', 'N/A')} (from ceiling)")
        print(f"   Confidence: {data.get('confidence_value', 'N/A')}")
        
        print_trust_flow(
            1.0, 
            data.get('trust_value', 0),
            "Trust capped by sensor.door.magnetic_switch ceiling"
        )
        
        return data['assertion_id']
    else:
        print(f"{Colors.FAIL}❌ Failed: {response.status_code} - {response.text}{Colors.ENDC}")
        return None

def create_sensor_assertion(sensor_id, temperature, confidence=0.98, token="sensor-token"):
    """Create a temperature sensor assertion - ROOT assertion"""
    print(f"\n{Colors.CYAN}📡 Creating Temperature Sensor Assertion{Colors.ENDC}")
    print(f"   Sensor ID: {sensor_id}")
    print(f"   Temperature: {temperature}°F")
    print(f"   Confidence: {confidence}")
    
    print_reasoning(
        "Root Sensor Trust",
        "As a root sensor, trust value is determined by the framework based on\n" +
        "   the sensor class ceiling. The sensor only reports its confidence in\n" +
        "   this specific reading."
    )
    
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
            "trust_explanation": "Temperature sensor calibrated and functioning normally"
        }
    )
    
    if response.status_code == 201:
        data = response.json()["data"]
        print(f"{Colors.GREEN}✅ Temperature sensor assertion created{Colors.ENDC}")
        print(f"   ID: {data['assertion_id']}")
        print(f"   Trust: {data.get('trust_value', 'N/A')} (from ceiling)")
        print(f"   Confidence: {data.get('confidence_value', 'N/A')}")
        
        print_trust_flow(
            1.0,
            data.get('trust_value', 0),
            "Trust set to sensor.temperature.honeywell_t7771a ceiling"
        )
        
        return data['assertion_id']
    else:
        print(f"{Colors.FAIL}❌ Failed: {response.status_code} - {response.text}{Colors.ENDC}")
        return None

def create_debug_assertion(content, token="debug-token"):
    """Create a debug/diagnostic assertion - typically consumed for context only"""
    print(f"\n{Colors.WARNING}🔧 Creating Debug Assertion{Colors.ENDC}")
    print(f"   Content: {json.dumps(content, indent=2)}")
    
    print_reasoning(
        "Debug Data Purpose",
        "Debug assertions provide context but typically shouldn't affect trust.\n" +
        "   They have a low trust ceiling (0.50) and consuming endpoints will\n" +
        "   likely mark them as context-only (apply_to_trust=False)."
    )
    
    response = requests.post(
        f"{BASE_URL}/api/v1/assertions",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        json={
            "endpoint_id": "debug-logger-01",
            "endpoint_class": "api.debug",
            "endpoint_type": "api",
            "content": content,
            "confidence": 1.0,
            "trust_explanation": "Debug/diagnostic data for context"
        }
    )
    
    if response.status_code == 201:
        data = response.json()["data"]
        print(f"{Colors.GREEN}✅ Debug assertion created{Colors.ENDC}")
        print(f"   ID: {data['assertion_id']}")
        print(f"   Trust: {data.get('trust_value', 'N/A')} (low ceiling for debug)")
        return data['assertion_id']
    else:
        print(f"{Colors.FAIL}❌ Failed: {response.status_code} - {response.text}{Colors.ENDC}")
        return None

def create_ml_assertion(sensor_ids, debug_id=None, token="ml-token"):
    """Create an ML model assertion with mixed trust/context inputs"""
    print(f"\n{Colors.BLUE}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BLUE}🤖 CREATING ML MODEL ASSERTION{Colors.ENDC}")
    print(f"{Colors.BLUE}{'='*70}{Colors.ENDC}")
    
    print_reasoning(
        "ML Model Trust Propagation",
        "This ML model consumes multiple inputs. Some affect trust (sensors),\n" +
        "   while others provide context (debug). The framework will calculate\n" +
        "   trust based on weighted propagation from trust-affecting inputs only."
    )
    
    # Get details of consumed assertions
    consumed_details = []
    all_consumed_ids = sensor_ids.copy()
    
    if debug_id:
        all_consumed_ids.append(debug_id)
    
    print(f"\n{Colors.BOLD}Fetching consumed assertion details...{Colors.ENDC}")
    for aid in all_consumed_ids:
        details = get_assertion_details(aid)
        if details:
            metadata = details['metadata']
            consumed_details.append({
                "id": aid,
                "trust_value": metadata.get("trust_value"),
                "trust_explanation": metadata.get("trust_explanation", "No explanation"),
                "content": details.get("content", {}),
                "is_debug": aid == debug_id
            })
            print(f"   • {aid}: Trust={metadata.get('trust_value', 'N/A')}")
    
    # Use AI to understand and determine materiality
    understanding = understand_trust_explanation(consumed_details, "ml_model")
    
    # Build materiality with tuple format [weight, apply_to_trust]
    materiality = {}
    
    print(f"\n{Colors.BOLD}Final Materiality Configuration:{Colors.ENDC}")
    for aid in all_consumed_ids:
        if aid in understanding['materiality_specs']:
            weight, apply = understanding['materiality_specs'][aid]
        elif aid == debug_id:
            weight, apply = 0.0, False
        else:
            weight = 1.0 / len(sensor_ids)
            apply = True
        
        materiality[aid] = [weight, apply]
        
        impact_str = f"Affects trust (weight={weight:.2%})" if apply else "Context only"
        color = Colors.GREEN if apply else Colors.WARNING
        print(f"   • {aid}: {color}{impact_str}{Colors.ENDC}")
    
    base_confidence = 0.85
    adjusted_confidence = base_confidence * understanding.get('confidence_impact', 1.0)
    
    print(f"\n{Colors.BOLD}Confidence Adjustment:{Colors.ENDC}")
    print(f"   Base: {base_confidence:.2f} → Adjusted: {adjusted_confidence:.2f}")
    print(f"   Reason: AI confidence impact factor = {understanding.get('confidence_impact', 1.0):.2f}")
    
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
                "analysis": "Based on sensor readings with debug context",
            },
            "confidence": adjusted_confidence,
            "trust_explanation": understanding['trust_explanation'],
            "consumed_assertions": all_consumed_ids,
            "consumed_assertion_materiality": materiality
        }
    )
    
    if response.status_code == 201:
        data = response.json()["data"]
        print(f"\n{Colors.GREEN}✅ ML assertion created successfully{Colors.ENDC}")
        print(f"   ID: {data['assertion_id']}")
        print(f"   Trust: {data.get('trust_value', 'N/A')} (propagated)")
        print(f"   Confidence: {data.get('confidence_value', 'N/A')}")
        print(f"   Trust Inputs: {data.get('trust_input_count', 'N/A')}")
        print(f"   Context Inputs: {data.get('context_input_count', 'N/A')}")
        
        # Explain trust propagation
        avg_sensor_trust = sum(c['trust_value'] for c in consumed_details if not c.get('is_debug')) / len(sensor_ids)
        print_trust_flow(
            avg_sensor_trust,
            data.get('trust_value', 0),
            f"Weighted propagation from {data.get('trust_input_count', 0)} trust inputs, " +
            f"capped by ml_model.predictive ceiling"
        )
        
        return data['assertion_id']
    else:
        print(f"{Colors.FAIL}❌ Failed: {response.status_code} - {response.text}{Colors.ENDC}")
        return None

def create_llm_assertion(ml_id, door_id, metadata_id=None, token="llm-token"):
    """Create an LLM assertion with both trust and context inputs"""
    print(f"\n{Colors.BLUE}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BLUE}🧠 CREATING LLM ASSERTION{Colors.ENDC}")
    print(f"{Colors.BLUE}{'='*70}{Colors.ENDC}")
    
    print_reasoning(
        "LLM Trust Synthesis",
        "The LLM combines ML predictions with sensor data. It uses AI to\n" +
        "   determine which inputs should affect trust vs. provide context.\n" +
        "   This creates the final recommendation with traceable trust."
    )
    
    # Get details of consumed assertions
    consumed_details = []
    all_consumed_ids = [ml_id, door_id]
    
    if metadata_id:
        all_consumed_ids.append(metadata_id)
    
    print(f"\n{Colors.BOLD}Gathering input assertions...{Colors.ENDC}")
    for aid in all_consumed_ids:
        details = get_assertion_details(aid)
        if details:
            metadata = details.get('metadata', {})
            consumed_details.append({
                "id": aid,
                "trust_value": metadata.get("trust_value"),
                "trust_explanation": metadata.get("trust_explanation", "No explanation"),
                "content": details.get("content", {}),
                "is_metadata": aid == metadata_id
            })
            content_preview = str(details.get("content", {}))[:50]
            print(f"   • {aid}: Trust={metadata.get('trust_value', 'N/A')}, Content={content_preview}...")
    
    # Use AI to synthesize and determine materiality
    understanding = understand_trust_explanation(consumed_details, "llm")
    
    # Build materiality specifications
    materiality = {}
    print(f"\n{Colors.BOLD}LLM Materiality Decisions:{Colors.ENDC}")
    
    for aid in all_consumed_ids:
        if aid in understanding['materiality_specs']:
            weight, apply = understanding['materiality_specs'][aid]
        elif aid == metadata_id:
            weight, apply = 0.0, False
        else:
            weight = 0.5 if aid == ml_id else 0.5
            apply = True
        
        materiality[aid] = [weight, apply]
        
        if apply:
            print(f"   • {aid}: {Colors.GREEN}Trust input (weight={weight:.1%}){Colors.ENDC}")
        else:
            print(f"   • {aid}: {Colors.WARNING}Context only{Colors.ENDC}")
    
    # Generate recommendation based on trust
    if understanding.get('confidence_impact', 1.0) < 0.7:
        recommendation = "Requires immediate human review due to trust concerns"
        risk_level = "high"
        print(f"\n{Colors.FAIL}⚠️  LOW TRUST DETECTED - Recommending human review{Colors.ENDC}")
    else:
        recommendation = "Schedule maintenance within 48 hours; ensure door is properly sealed"
        risk_level = "medium"
        print(f"\n{Colors.GREEN}✓ Sufficient trust - Automated recommendation generated{Colors.ENDC}")
    
    print_reasoning(
        "Final Recommendation Logic",
        f"Based on propagated trust and confidence, the LLM determined:\n" +
        f"   Risk Level: {risk_level}\n" +
        f"   Action: {recommendation}"
    )
    
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
                "analysis": "Comprehensive analysis with trust and context inputs",
                "timestamp": datetime.utcnow().isoformat()
            },
            "confidence": 0.88 * understanding.get('confidence_impact', 1.0),
            "trust_explanation": understanding['trust_explanation'],
            "consumed_assertions": all_consumed_ids,
            "consumed_assertion_materiality": materiality
        }
    )
    
    if response.status_code == 201:
        data = response.json()["data"]
        print(f"\n{Colors.GREEN}✅ LLM assertion created successfully{Colors.ENDC}")
        print(f"   ID: {data['assertion_id']}")
        print(f"   Trust: {data.get('trust_value', 'N/A')} (propagated)")
        print(f"   Confidence: {data.get('confidence_value', 'N/A')}")
        print(f"   Trust Inputs: {data.get('trust_input_count', 'N/A')}")
        print(f"   Context Inputs: {data.get('context_input_count', 'N/A')}")
        
        # Show final trust flow
        ml_trust = next(c['trust_value'] for c in consumed_details if c['id'] == ml_id)
        door_trust = next(c['trust_value'] for c in consumed_details if c['id'] == door_id)
        avg_input_trust = (ml_trust + door_trust) / 2
        
        print_trust_flow(
            avg_input_trust,
            data.get('trust_value', 0),
            f"Combined ML ({ml_trust:.3f}) and door ({door_trust:.3f}) trust, " +
            f"capped by llm.gpt4 ceiling"
        )
        
        return data['assertion_id']
    else:
        print(f"{Colors.FAIL}❌ Failed: {response.status_code} - {response.text}{Colors.ENDC}")
        return None

def get_evidence(assertion_id, token="user-token"):
    """Get and display the evidence tree for an assertion"""
    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}🔍 EVIDENCE CHAIN ANALYSIS{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")
    
    print_reasoning(
        "Evidence Retrieval",
        "Fetching the complete chain of assertions that led to this conclusion.\n" +
        "   This shows how trust propagated through the system."
    )
    
    response = requests.get(
        f"{BASE_URL}/api/v1/assertions/{assertion_id}/evidence",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    if response.status_code == 200:
        data = response.json()["data"]
        summary = data["summary"]
        
        print(f"\n{Colors.BOLD}Chain Statistics:{Colors.ENDC}")
        print(f"   📊 Chain Depth: {summary['chain_depth']} levels")
        print(f"   📊 Total Assertions: {summary['total_assertions']}")
        print(f"   📊 Trust-Affecting: {summary.get('trust_inputs', 'N/A')}")
        print(f"   📊 Context-Only: {summary.get('context_inputs', 'N/A')}")
        
        if summary.get("weakest_link"):
            weakest = summary["weakest_link"]
            print(f"\n{Colors.WARNING}⚠️  WEAKEST LINK IDENTIFIED:{Colors.ENDC}")
            print(f"   Endpoint: {weakest['endpoint_id']}")
            print(f"   Trust Value: {Colors.FAIL}{weakest['trust_value']}{Colors.ENDC}")
            print(f"   Explanation: {weakest.get('explanation', 'N/A')}")
            
            print_reasoning(
                "Weakest Link Impact",
                "This is the point in the chain with the lowest trust.\n" +
                "   It represents the primary bottleneck for overall system trust.\n" +
                "   Improving this component would have the greatest impact."
            )
        
        print(f"\n{Colors.BOLD}💡 System Recommendation:{Colors.ENDC} {summary['recommendation']}")
        
        return data
    else:
        print(f"{Colors.FAIL}❌ Failed to get evidence: {response.status_code} - {response.text}{Colors.ENDC}")
        return None

def get_trust_matrix(assertion_id, token="user-token"):
    """Get trust-confidence matrix classification"""
    print(f"\n{Colors.HEADER}📊 TRUST-CONFIDENCE MATRIX{Colors.ENDC}")
    
    response = requests.get(
        f"{BASE_URL}/api/v1/assertions/{assertion_id}/matrix",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    if response.status_code == 200:
        data = response.json()["data"]
        
        # Visual matrix representation
        trust = data['trust_value']
        confidence = data['confidence_value']
        
        print(f"\n   High │ {Colors.WARNING}Low Trust{Colors.ENDC}      │ {Colors.GREEN}High Trust{Colors.ENDC}")
        print(f"   Conf │ {Colors.WARNING}High Confidence{Colors.ENDC} │ {Colors.GREEN}High Confidence{Colors.ENDC}")
        print(f"        │ {'⚠️ CAUTION' if trust < 0.7 and confidence >= 0.7 else '            '} │ {'✅ PROCEED' if trust >= 0.7 and confidence >= 0.7 else '           '}")
        print(f"        ├────────────────┼─────────────────┤")
        print(f"   Low  │ {Colors.FAIL}Low Trust{Colors.ENDC}       │ {Colors.CYAN}High Trust{Colors.ENDC}")
        print(f"   Conf │ {Colors.FAIL}Low Confidence{Colors.ENDC}  │ {Colors.CYAN}Low Confidence{Colors.ENDC}")
        print(f"        │ {'❌ DEFER' if trust < 0.7 and confidence < 0.7 else '          '}   │ {'🤔 REVIEW' if trust >= 0.7 and confidence < 0.7 else '          '}")
        print(f"        └────────────────┴─────────────────┘")
        print(f"           Low Trust        High Trust")
        
        print(f"\n{Colors.BOLD}Current Position:{Colors.ENDC}")
        print(f"   Trust: {trust:.2f}, Confidence: {confidence:.2f}")
        print(f"   Status: {data['status'].replace('_', ' ').title()}")
        print(f"   Action: {data['recommendation']}")
        
        print_reasoning(
            "Matrix Interpretation",
            "The trust-confidence matrix helps determine appropriate actions:\n" +
            "   • High Trust + High Confidence = Automated action safe\n" +
            "   • High Trust + Low Confidence = Human judgment needed\n" +
            "   • Low Trust + High Confidence = Dangerous overconfidence\n" +
            "   • Low Trust + Low Confidence = Need more information"
        )
        
        return data
    else:
        print(f"{Colors.FAIL}❌ Failed to get matrix: {response.status_code} - {response.text}{Colors.ENDC}")
        return None

def main():
    """Main execution flow demonstrating tuple-based materiality"""
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}AI TRUST CHAIN FRAMEWORK DEMONSTRATION{Colors.ENDC}")
    print(f"{Colors.HEADER}Trust Propagation with Intelligent Materiality{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")
    
    # Check Azure OpenAI configuration
    has_azure = bool(AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT and 
                     not AZURE_OPENAI_ENDPOINT.startswith("https://your-resource"))
    
    if not has_azure:
        print(f"\n{Colors.WARNING}⚠️  WARNING: Azure OpenAI not configured!{Colors.ENDC}")
        print("Current environment variables:")
        print(f"  AZURE_OPENAI_ENDPOINT: {AZURE_OPENAI_ENDPOINT if AZURE_OPENAI_ENDPOINT else 'Not set'}")
        print(f"  AZURE_OPENAI_KEY: {'Set' if AZURE_OPENAI_KEY else 'Not set'}")
        print(f"  AZURE_OPENAI_DEPLOYMENT: {AZURE_OPENAI_DEPLOYMENT}")
        print("\nTo enable AI-enhanced understanding, set:")
        print("  export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com'")
        print("  export AZURE_OPENAI_KEY='your-api-key'")
        print("  export AZURE_OPENAI_DEPLOYMENT='your-deployment-name'")
        print(f"\n{Colors.BOLD}Continuing with rule-based fallback mode...{Colors.ENDC}\n")
        time.sleep(2)
    else:
        print(f"\n{Colors.GREEN}✅ Azure OpenAI configured{Colors.ENDC}")
        print(f"  Endpoint: {AZURE_OPENAI_ENDPOINT}")
        print(f"  Deployment: {AZURE_OPENAI_DEPLOYMENT}")
        print(f"  API Version: {AZURE_API_VERSION}")
    
    # Step 1: Initialize system
    print(f"\n{Colors.HEADER}STEP 1: SYSTEM INITIALIZATION{Colors.ENDC}")
    print("-"*40)
    initialize_system()
    
    # Step 2: Set trust ceilings
    print(f"\n{Colors.HEADER}STEP 2: CONFIGURING TRUST CEILINGS{Colors.ENDC}")
    print("-"*40)
    
    print_reasoning(
        "Trust Ceiling Strategy",
        "Different endpoint types have different maximum trust levels.\n" +
        "   This reflects their inherent reliability and potential for error.\n" +
        "   Sensors > ML Models > LLMs in terms of trust ceilings."
    )
    
    set_trust_ceiling("sensor.temperature.honeywell_t7771a", 0.90)
    set_trust_ceiling("sensor.door.magnetic_switch", 0.95)
    set_trust_ceiling("api.debug", 0.50)
    set_trust_ceiling("ml_model.predictive", 0.85)
    set_trust_ceiling("llm.gpt4", 0.80)
    
    # Step 3: Create assertion chain
    print(f"\n{Colors.HEADER}STEP 3: BUILDING ASSERTION CHAIN{Colors.ENDC}")
    print("-"*40)
    
    print_reasoning(
        "Chain Construction",
        "We'll build a chain: Sensors → ML Model → LLM\n" +
        "   Each level will consume assertions from the previous level.\n" +
        "   Trust will propagate and degrade based on the weakest links."
    )
    
    # Create temperature sensors
    print(f"\n{Colors.BOLD}Creating Temperature Sensors (Trust Inputs)...{Colors.ENDC}")
    sensor1_id = create_sensor_assertion("temp-sensor-01", 75.2, 0.98)
    if not sensor1_id:
        print("Failed to create sensor 1. Exiting.")
        sys.exit(1)
    
    sensor2_id = create_sensor_assertion("temp-sensor-02", 76.8, 0.95)
    if not sensor2_id:
        print("Failed to create sensor 2. Exiting.")
        sys.exit(1)
    
    # Create door sensor
    print(f"\n{Colors.BOLD}Creating Door Sensor (Trust Input)...{Colors.ENDC}")
    door_id = create_door_sensor_assertion("fridge-door-01", is_open=True, confidence=0.99)
    if not door_id:
        print("Failed to create door sensor. Exiting.")
        sys.exit(1)
    
    # Create debug assertion
    print(f"\n{Colors.BOLD}Creating Debug Data (Context Only)...{Colors.ENDC}")
    debug_id = create_debug_assertion({
        "system_load": 0.45,
        "memory_usage": "2.3GB",
        "last_calibration": "2024-01-15",
        "diagnostics": "All systems nominal"
    })
    
    # Create ML model consuming sensors + debug
    print(f"\n{Colors.BOLD}Creating ML Model (Consumes Sensors + Debug)...{Colors.ENDC}")
    ml_id = create_ml_assertion([sensor1_id, sensor2_id], debug_id)
    if not ml_id:
        print("Failed to create ML assertion. Exiting.")
        sys.exit(1)
    
    # Create metadata assertion for LLM context
    print(f"\n{Colors.BOLD}Creating Metadata (Context for LLM)...{Colors.ENDC}")
    metadata_id = create_debug_assertion({
        "facility": "Building A",
        "room": "Server Room 3",
        "maintenance_history": "Last serviced 2024-12-01"
    })
    
    # Create LLM consuming ML + door + metadata
    print(f"\n{Colors.BOLD}Creating LLM (Final Analysis)...{Colors.ENDC}")
    llm_id = create_llm_assertion(ml_id, door_id, metadata_id)
    if not llm_id:
        print("Failed to create LLM assertion. Exiting.")
        sys.exit(1)
    
    # Step 4: Display evidence
    print(f"\n{Colors.HEADER}STEP 4: ANALYZING EVIDENCE CHAIN{Colors.ENDC}")
    print("-"*40)
    evidence = get_evidence(llm_id)
    
    # Step 5: Get trust matrix
    print(f"\n{Colors.HEADER}STEP 5: TRUST-CONFIDENCE MATRIX{Colors.ENDC}")
    print("-"*40)
    matrix = get_trust_matrix(llm_id)
    
    # Final summary
    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}DEMONSTRATION COMPLETE{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")
    
    print(f"\n{Colors.GREEN}🎯 Final LLM Assertion ID: {llm_id}{Colors.ENDC}")
    
    print(f"\n{Colors.BOLD}📊 Key Insights Learned:{Colors.ENDC}")
    print("   1. Trust values are NEVER self-reported - always calculated")
    print("   2. Trust propagates through weighted averages of trust inputs")
    print("   3. Context inputs provide information but don't affect trust")
    print("   4. Each endpoint class has a trust ceiling it cannot exceed")
    print("   5. The weakest link in the chain limits overall trust")
    print("   6. AI can intelligently determine materiality of inputs")
    
    print(f"\n{Colors.CYAN}You can query assertion {llm_id} for more details.{Colors.ENDC}")
    print(f"{Colors.BOLD}The entire chain is immutably stored in the blockchain audit trail.{Colors.ENDC}")

if __name__ == "__main__":
    main()
