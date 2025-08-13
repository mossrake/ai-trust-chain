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

def generate_llm_assertion_content(consumed_details: List[Dict]) -> Dict:
    """
    Use Azure OpenAI to generate the actual LLM assertion content
    by synthesizing the consumed assertions into a recommendation
    """
    print(f"\n{Colors.CYAN}🤖 LLM CONTENT GENERATION{Colors.ENDC}")
    print(f"   Synthesizing {len(consumed_details)} inputs into recommendation...")
    
    if not AZURE_OPENAI_KEY or AZURE_OPENAI_KEY == "your-api-key":
        print(f"   {Colors.WARNING}⚠️  Azure OpenAI not configured - using fallback content{Colors.ENDC}")
        
        # Fallback: Create basic content from consumed data
        door_status = "unknown"
        ml_prediction = "unknown"
        temp_readings = []
        
        for detail in consumed_details:
            content = detail.get('content', {})
            if 'door' in detail['id'].lower():
                door_status = "open" if content.get('is_open', False) else "closed"
            elif 'ml-model' in detail['id']:
                ml_prediction = content.get('prediction', 'unknown')
            elif 'temp-sensor' in detail['id']:
                if 'temperature' in content:
                    temp_readings.append(content['temperature'])
        
        if door_status == "open":
            recommendation = (
                f"URGENT: Door sensor reports OPEN status. "
                f"Immediate action required to prevent temperature excursion. "
                f"Current temperatures: {temp_readings}°F. "
                f"ML model indicates: {ml_prediction}."
            )
            risk_level = "high"
        else:
            recommendation = (
                f"Door is {door_status}. Temperatures at {temp_readings}°F. "
                f"ML model prediction: {ml_prediction}. "
                f"Monitor situation and schedule maintenance if patterns persist."
            )
            risk_level = "medium" if ml_prediction == "maintenance_required" else "low"
        
        return {
            "recommendation": recommendation,
            "risk_level": risk_level,
            "analysis": "Fallback analysis without AI synthesis",
            "action_items": ["Review sensor data", "Take appropriate action"]
        }
    
    # Build context for LLM to synthesize
    print(f"\n   {Colors.BOLD}Preparing synthesis context...{Colors.ENDC}")
    
    # Prepare consumed data for the prompt
    consumed_data = []
    for d in consumed_details:
        consumed_data.append({
            'source': d['id'],
            'content': d['content'],
            'trust_level': d.get('trust_value', 'N/A'),
            'trust_explanation': d.get('trust_explanation', '')
        })
    
    llm_prompt = f"""You are an LLM endpoint in a trust chain system that needs to synthesize multiple data sources into an actionable recommendation.

Your role is to analyze the actual values from consumed assertions and create a specific, actionable recommendation based on what you observe.

Consumed assertions:
{json.dumps(consumed_data, indent=2)}

Instructions:
1. Analyze the ACTUAL VALUES in the consumed content (temperatures, door status, predictions, etc.)
2. Synthesize these into a SPECIFIC recommendation that addresses what you see
3. Don't give generic advice - be specific about the actual conditions observed
4. If a door is open, say it's open. If temperatures are high, state the actual values
5. Provide concrete action steps based on the current state

Respond with JSON containing:
{json.dumps({
    "recommendation": "A specific, detailed recommendation based on the actual data you see above",
    "risk_level": "high/medium/low based on actual conditions",
    "analysis": "Your synthesis of what the consumed data shows",
    "action_items": ["Specific action 1", "Specific action 2", "..."],
    "key_observations": ["What you actually found in the data"]
}, indent=2)}

Remember: Be specific about what you see in the data, not generic."""
    
    print(f"   {Colors.BOLD}Calling Azure OpenAI for content synthesis...{Colors.ENDC}")
    
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_KEY
    }
    
    try:
        response = requests.post(
            f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}",
            headers=headers,
            json={
                "messages": [
                    {"role": "system", "content": "You are an intelligent LLM endpoint that synthesizes sensor data, ML predictions, and other inputs into specific, actionable recommendations. Always reference the actual values you see in the data."},
                    {"role": "user", "content": llm_prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 600
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Check if we got a valid response
            if 'choices' not in result or len(result['choices']) == 0:
                print(f"   {Colors.WARNING}⚠️  Unexpected response format from Azure OpenAI{Colors.ENDC}")
                print(f"   Response: {json.dumps(result, indent=2)[:500]}")
                return {
                    "recommendation": "Unable to generate AI recommendation - unexpected response format",
                    "risk_level": "unknown",
                    "analysis": "AI response format error",
                    "action_items": ["Manual review required"]
                }
            
            llm_response = result['choices'][0]['message']['content']
            
            # Debug: Show what we got
            #print(f"   {Colors.BOLD}Raw AI Response (first 200 chars):{Colors.ENDC}")
            #print(f"   {llm_response[:200]}...")
            
            # Clean up the response - remove markdown code blocks if present
            cleaned_response = llm_response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]  # Remove ```json
            elif cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]  # Remove ```
            
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]  # Remove trailing ```
            
            cleaned_response = cleaned_response.strip()
            
            try:
                # Try to parse the cleaned JSON response
                llm_content = json.loads(cleaned_response)
                
                print(f"\n   {Colors.GREEN}✅ LLM Content Generated:{Colors.ENDC}")
                print(f"   {Colors.BOLD}Risk Level:{Colors.ENDC} {llm_content.get('risk_level', 'unknown')}")
                print(f"   {Colors.BOLD}Key Observations:{Colors.ENDC}")
                for obs in llm_content.get('key_observations', [])[:3]:
                    print(f"     • {obs}")
                print(f"   {Colors.BOLD}Recommendation Preview:{Colors.ENDC}")
                print(f"     {llm_content.get('recommendation', '')[:150]}...")
                
                return llm_content
                
            except json.JSONDecodeError as e:
                print(f"   {Colors.WARNING}⚠️  Could not parse LLM response as JSON{Colors.ENDC}")
                print(f"   JSON Error: {str(e)}")
                print(f"   Cleaned response (first 300 chars): {cleaned_response[:300]}")
                
                # Try to extract meaningful content from the response
                if cleaned_response and len(cleaned_response) > 0:
                    # Use the raw response as the recommendation if it's text
                    return {
                        "recommendation": cleaned_response[:500],
                        "risk_level": "medium",
                        "analysis": "AI provided text response instead of JSON",
                        "action_items": ["Review the text recommendation above", "Take appropriate action"],
                        "key_observations": ["Response was not in expected JSON format"]
                    }
                else:
                    return {
                        "recommendation": "Failed to generate AI recommendation - empty response",
                        "risk_level": "unknown",
                        "analysis": "Empty AI response",
                        "action_items": ["Manual review required"]
                    }
        else:
            print(f"   {Colors.FAIL}❌ Azure OpenAI call failed: {response.status_code}{Colors.ENDC}")
            return {
                "recommendation": "Failed to generate AI recommendation - manual review required",
                "risk_level": "unknown",
                "analysis": "AI synthesis unavailable",
                "action_items": ["Review consumed assertions manually"]
            }
            
    except Exception as e:
        print(f"   {Colors.FAIL}❌ Error calling Azure OpenAI: {e}{Colors.ENDC}")
        return {
            "recommendation": "Error generating recommendation - manual review required",
            "risk_level": "unknown",
            "analysis": f"Error: {str(e)}",
            "action_items": ["Check system configuration", "Review data manually"]
        }

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
        "trust_explanation": "your detailed trust status explanation, naming the endpoints by their instance label"
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
                print(f"   {Colors.BOLD}Understanding:{Colors.ENDC} {ai_result.get('understanding', '')}...")
                
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
    print(f"\n{Colors.BLUE}🔐 Setting Trust Ceiling{Colors.ENDC}")
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
        "   determine which inputs should affect trust vs. provide context,\n" +
        "   then synthesizes the actual data into a specific recommendation."
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
    
    # Use AI to understand trust and determine materiality
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
    
    # Generate the actual LLM assertion content by synthesizing consumed data
    llm_content = generate_llm_assertion_content(consumed_details)
    
    # Adjust confidence based on trust understanding
    base_confidence = 0.88
    adjusted_confidence = base_confidence * understanding.get('confidence_impact', 1.0)
    
    print(f"\n{Colors.BOLD}Final LLM Assertion:{Colors.ENDC}")
    print(f"   Risk Level: {llm_content.get('risk_level', 'unknown')}")
    print(f"   Confidence: {adjusted_confidence:.2f}")
    
    print_reasoning(
        "Final Recommendation Logic",
        f"The LLM synthesized the actual data from its inputs:\n" +
        f"   - ML prediction, door status, temperature readings\n" +
        f"   - Generated specific recommendations based on observed values\n" +
        f"   - Risk Level: {llm_content.get('risk_level', 'unknown')}"
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
                "recommendation": llm_content.get('recommendation', 'Failed to generate recommendation'),
                "risk_level": llm_content.get('risk_level', 'unknown'),
                "analysis": llm_content.get('analysis', 'LLM synthesis of consumed assertions'),
                "action_items": llm_content.get('action_items', []),
                "key_observations": llm_content.get('key_observations', []),
                "timestamp": datetime.utcnow().isoformat()
            },
            "confidence": adjusted_confidence,
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
        ml_trust = next((c['trust_value'] for c in consumed_details if c['id'] == ml_id), 0)
        door_trust = next((c['trust_value'] for c in consumed_details if c['id'] == door_id), 0)
        avg_input_trust = (ml_trust + door_trust) / 2 if ml_trust and door_trust else 0
        
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
    
    #print(f"\n{Colors.BOLD}📊 Key Insights Learned:{Colors.ENDC}")
    #print("   1. Trust values are NEVER self-reported - always calculated")
    #print("   2. Trust propagates through weighted averages of trust inputs")
    #print("   3. Context inputs provide information but don't affect trust")
    #print("   4. Each endpoint class has a trust ceiling it cannot exceed")
    #print("   5. The weakest link in the chain limits overall trust")
    #print("   6. AI can intelligently determine materiality of inputs")
    #print("   7. LLM endpoints synthesize actual data into specific recommendations")
    
    print(f"\n{Colors.CYAN}You can query assertion {llm_id} for more details.{Colors.ENDC}")
    print(f"{Colors.BOLD}The entire chain is immutably stored in the blockchain audit trail.{Colors.ENDC}")

if __name__ == "__main__":
    main()
