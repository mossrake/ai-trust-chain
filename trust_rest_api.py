"""
AI Trust Chain Framework - REST API Server
Copyright (C) 2025 Mossrake Group, LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This module provides RESTful API endpoints for trust management, administration,
and assertion processing with pass-through authentication.

Based on the AI Trust Chain Framework designed by Mossrake Group, LLC.
The framework design and concepts are proprietary intellectual property
of Mossrake Group, LLC. This implementation is released under AGPL-3.0.
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from functools import wraps
from typing import Dict, List, Optional, Any

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from werkzeug.exceptions import BadRequest, Unauthorized, NotFound

from trust_core_kernel import (
    TrustAuthority, BlockchainAuditTrail, TrustKernel,
    TrustRule, Assertion, TrustMetadata, EndpointType
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['TRUST_REGISTRY_PATH'] = os.environ.get('TRUST_REGISTRY_PATH', 'trust_registry.json')
app.config['BLOCKCHAIN_DB_PATH'] = os.environ.get('BLOCKCHAIN_DB_PATH', 'trust_chain.db')

# Initialize trust system components
trust_authority = TrustAuthority(app.config['TRUST_REGISTRY_PATH'])
audit_trail = BlockchainAuditTrail(app.config['BLOCKCHAIN_DB_PATH'])
trust_kernel = TrustKernel(trust_authority, audit_trail)

@dataclass
class ApiResponse:
    """Standard API response structure"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().timestamp()
    
    def to_json(self) -> Dict:
        return {
            'success': self.success,
            'data': self.data,
            'error': self.error,
            'timestamp': self.timestamp
        }

def pass_through_auth(f):
    """
    Pass-through authentication decorator.
    In production, replace with proper authentication/authorization.
    Currently accepts any Authorization header for demonstration.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify(ApiResponse(
                success=False,
                error="Authorization header required"
            ).to_json()), 401
        
        # Pass-through: accept any auth header
        # In production, validate against auth service
        request.auth_user = auth_header.replace('Bearer ', '')
        return f(*args, **kwargs)
    
    return decorated_function

def admin_required(f):
    """Admin authorization decorator"""
    @wraps(f)
    @pass_through_auth
    def decorated_function(*args, **kwargs):
        # Simple admin check - in production, check against proper roles
        if not request.auth_user.startswith('admin'):
            return jsonify(ApiResponse(
                success=False,
                error="Admin privileges required"
            ).to_json()), 403
        return f(*args, **kwargs)
    
    return decorated_function

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify(ApiResponse(
        success=True,
        data={
            'status': 'healthy',
            'chain_integrity': audit_trail.verify_chain_integrity()
        }
    ).to_json())

# Trust Registry Management Endpoints
@app.route('/api/v1/trust/registry', methods=['GET'])
@pass_through_auth
def get_trust_registry():
    """Get current trust registry configuration"""
    return jsonify(ApiResponse(
        success=True,
        data={
            'trust_registry': trust_authority.trust_registry,
            'propagation_rules': {
                rule_id: asdict(rule) 
                for rule_id, rule in trust_authority.propagation_rules.items()
            }
        }
    ).to_json())

@app.route('/api/v1/trust/endpoint-class', methods=['POST'])
@admin_required
def set_endpoint_trust():
    """Set trust ceiling for an endpoint class"""
    data = request.get_json()
    
    if not data or 'endpoint_class' not in data or 'max_trust' not in data:
        return jsonify(ApiResponse(
            success=False,
            error="endpoint_class and max_trust required"
        ).to_json()), 400
    
    try:
        trust_authority.set_endpoint_trust_ceiling(
            data['endpoint_class'],
            float(data['max_trust'])
        )
        return jsonify(ApiResponse(
            success=True,
            data={'message': f"Trust ceiling set for {data['endpoint_class']}"}
        ).to_json())
    except ValueError as e:
        return jsonify(ApiResponse(
            success=False,
            error=str(e)
        ).to_json()), 400

@app.route('/api/v1/trust/propagation-rule', methods=['POST'])
@admin_required
def add_propagation_rule():
    """Add or update a trust propagation rule"""
    data = request.get_json()
    
    required_fields = ['rule_id', 'endpoint_type', 'max_trust_ceiling', 'propagation_method']
    if not all(field in data for field in required_fields):
        return jsonify(ApiResponse(
            success=False,
            error=f"Required fields: {required_fields}"
        ).to_json()), 400
    
    try:
        # Convert endpoint_type string to enum
        endpoint_type = EndpointType(data['endpoint_type'])
        
        rule = TrustRule(
            rule_id=data['rule_id'],
            endpoint_type=endpoint_type,
            max_trust_ceiling=float(data['max_trust_ceiling']),
            propagation_method=data['propagation_method'],
            consensus_threshold=data.get('consensus_threshold'),
            weight_factors=data.get('weight_factors', {})
        )
        
        trust_authority.add_propagation_rule(rule)
        
        return jsonify(ApiResponse(
            success=True,
            data={'message': f"Propagation rule {data['rule_id']} added/updated"}
        ).to_json())
    except (ValueError, KeyError) as e:
        return jsonify(ApiResponse(
            success=False,
            error=str(e)
        ).to_json()), 400

@app.route('/api/v1/trust/propagation-rule/<rule_id>', methods=['DELETE'])
@admin_required
def delete_propagation_rule(rule_id):
    """Delete a trust propagation rule"""
    if rule_id in trust_authority.propagation_rules:
        del trust_authority.propagation_rules[rule_id]
        trust_authority.save_registry()
        return jsonify(ApiResponse(
            success=True,
            data={'message': f"Propagation rule {rule_id} deleted"}
        ).to_json())
    else:
        return jsonify(ApiResponse(
            success=False,
            error=f"Rule {rule_id} not found"
        ).to_json()), 404

# Assertion Management Endpoints
@app.route('/api/v1/assertions', methods=['POST'])
@pass_through_auth
def create_assertion():
    """
    Create a new assertion with trust evaluation
    
    Request body should include:
    - endpoint_id: Unique identifier for the endpoint
    - endpoint_class: Class for trust ceiling lookup
    - endpoint_type: Type of endpoint (sensor, ml_model, llm, etc.)
    - content: The assertion data
    - confidence: Self-reported confidence (0.0-1.0)
    - consumed_assertions: List of assertion IDs this is based on (optional)
    - limitations: Any limitations or caveats (optional)
    
    Note: Materiality of consumed assertions is determined internally
    by the endpoint based on its logic, not passed via API.
    """
    data = request.get_json()
    
    required_fields = ['endpoint_id', 'endpoint_class', 'endpoint_type', 'content', 'confidence']
    if not all(field in data for field in required_fields):
        return jsonify(ApiResponse(
            success=False,
            error=f"Required fields: {required_fields}"
        ).to_json()), 400
    
    try:
        endpoint_type = EndpointType(data['endpoint_type'])
        
        assertion = trust_kernel.create_assertion(
            endpoint_id=data['endpoint_id'],
            endpoint_class=data['endpoint_class'],
            endpoint_type=endpoint_type,
            content=data['content'],
            confidence=float(data['confidence']),
            consumed_assertion_ids=data.get('consumed_assertions', []),
            limitations=data.get('limitations', {})
        )
        
        return jsonify(ApiResponse(
            success=True,
            data={
                'assertion_id': assertion.id,
                'trust_value': assertion.metadata.trust_value,
                'confidence_value': assertion.metadata.confidence_value,
                'temporal_validity': assertion.metadata.temporal_validity,
                'trust_explanation': assertion.metadata.trust_explanation
            }
        ).to_json()), 201
    except (ValueError, KeyError) as e:
        return jsonify(ApiResponse(
            success=False,
            error=str(e)
        ).to_json()), 400

@app.route('/api/v1/assertions/<assertion_id>', methods=['GET'])
@pass_through_auth
def get_assertion(assertion_id):
    """Retrieve an assertion by ID"""
    assertion = trust_kernel.get_assertion(assertion_id)
    
    if not assertion:
        return jsonify(ApiResponse(
            success=False,
            error=f"Assertion {assertion_id} not found"
        ).to_json()), 404
    
    return jsonify(ApiResponse(
        success=True,
        data=assertion.to_dict()
    ).to_json())

@app.route('/api/v1/assertions/<assertion_id>/provenance', methods=['GET'])
@pass_through_auth
def get_assertion_provenance(assertion_id):
    """Get complete provenance chain for an assertion (legacy endpoint)"""
    provenance = trust_kernel.get_assertion_provenance(assertion_id)
    
    if not provenance['chain']:
        return jsonify(ApiResponse(
            success=False,
            error=f"Assertion {assertion_id} not found"
        ).to_json()), 404
    
    return jsonify(ApiResponse(
        success=True,
        data=provenance
    ).to_json())

@app.route('/api/v1/assertions/<assertion_id>/evidence', methods=['GET'])
@pass_through_auth
def show_evidence(assertion_id):
    """
    'Show me the evidence' endpoint - returns complete evidence chain
    
    This is the key endpoint for understanding how an AI recommendation
    was derived, showing all assertions that contributed to the final result.
    """
    evidence = trust_kernel.show_evidence_chain(assertion_id)
    
    if "error" in evidence.get("evidence_tree", {}):
        return jsonify(ApiResponse(
            success=False,
            error=evidence["evidence_tree"]["error"]
        ).to_json()), 404
    
    return jsonify(ApiResponse(
        success=True,
        data=evidence
    ).to_json())

@app.route('/api/v1/assertions/<assertion_id>/explain', methods=['GET'])
@pass_through_auth
def explain_assertion(assertion_id):
    """
    Get human-readable explanation of the assertion evidence chain
    
    Query parameters:
    - format: 'text' (default) or 'markdown'
    """
    format_type = request.args.get('format', 'text')
    
    if format_type not in ['text', 'markdown']:
        return jsonify(ApiResponse(
            success=False,
            error="Format must be 'text' or 'markdown'"
        ).to_json()), 400
    
    try:
        explanation = trust_kernel.explain_assertion_chain(assertion_id, format_type)
        
        return jsonify(ApiResponse(
            success=True,
            data={
                'assertion_id': assertion_id,
                'format': format_type,
                'explanation': explanation
            }
        ).to_json())
    except Exception as e:
        return jsonify(ApiResponse(
            success=False,
            error=str(e)
        ).to_json()), 500

@app.route('/api/v1/assertions/<assertion_id>/matrix', methods=['GET'])
@pass_through_auth
def get_trust_confidence_matrix(assertion_id):
    """Get trust-confidence matrix classification for an assertion"""
    matrix = trust_kernel.get_trust_confidence_matrix(assertion_id)
    
    if matrix.get('status') == 'unknown':
        return jsonify(ApiResponse(
            success=False,
            error=f"Assertion {assertion_id} not found"
        ).to_json()), 404
    
    return jsonify(ApiResponse(
        success=True,
        data=matrix
    ).to_json())

# Blockchain Management Endpoints
@app.route('/api/v1/blockchain/commit', methods=['POST'])
@pass_through_auth
def commit_assertions():
    """Manually commit pending assertions to blockchain"""
    block = trust_kernel.commit_pending_assertions()
    
    if not block:
        return jsonify(ApiResponse(
            success=True,
            data={'message': 'No pending assertions to commit'}
        ).to_json())
    
    return jsonify(ApiResponse(
        success=True,
        data={
            'block_index': block.index,
            'block_hash': block.hash,
            'assertions_count': len(block.assertions)
        }
    ).to_json())

@app.route('/api/v1/blockchain/verify', methods=['GET'])
@pass_through_auth
def verify_blockchain():
    """Verify blockchain integrity"""
    is_valid = audit_trail.verify_chain_integrity()
    
    return jsonify(ApiResponse(
        success=True,
        data={
            'integrity_valid': is_valid,
            'latest_block': {
                'index': audit_trail.get_latest_block().index,
                'hash': audit_trail.get_latest_block().hash
            } if audit_trail.get_latest_block() else None
        }
    ).to_json())

@app.route('/api/v1/blockchain/latest', methods=['GET'])
@pass_through_auth
def get_latest_block():
    """Get information about the latest block"""
    block = audit_trail.get_latest_block()
    
    if not block:
        return jsonify(ApiResponse(
            success=False,
            error="No blocks found"
        ).to_json()), 404
    
    return jsonify(ApiResponse(
        success=True,
        data={
            'index': block.index,
            'hash': block.hash,
            'timestamp': block.timestamp,
            'assertions_count': len(block.assertions),
            'previous_hash': block.previous_hash
        }
    ).to_json())

# Admin Initialization Endpoint
@app.route('/api/v1/admin/initialize', methods=['POST'])
@admin_required
def initialize_system():
    """Initialize trust system with default configuration"""
    data = request.get_json()
    
    # Set default trust ceilings
    default_ceilings = data.get('default_ceilings', {
        'sensor.temperature.honeywell_t7771a': 0.85,
        'sensor.generic': 0.7,
        'api.external': 0.6,
        'ml_model.predictive': 0.75,
        'llm.gpt4': 0.8,
        'llm.generic': 0.65
    })
    
    for endpoint_class, max_trust in default_ceilings.items():
        trust_authority.set_endpoint_trust_ceiling(endpoint_class, max_trust)
    
    # Add default propagation rules
    default_rules = [
        TrustRule(
            rule_id='sensor_minimum',
            endpoint_type=EndpointType.SENSOR,
            max_trust_ceiling=0.9,
            propagation_method='minimum'
        ),
        TrustRule(
            rule_id='ml_weighted',
            endpoint_type=EndpointType.ML_MODEL,
            max_trust_ceiling=0.85,
            propagation_method='weighted_average'
        ),
        TrustRule(
            rule_id='llm_consensus',
            endpoint_type=EndpointType.LLM,
            max_trust_ceiling=0.8,
            propagation_method='consensus',
            consensus_threshold=0.7
        )
    ]
    
    for rule in default_rules:
        trust_authority.add_propagation_rule(rule)
    
    return jsonify(ApiResponse(
        success=True,
        data={
            'message': 'System initialized with default configuration',
            'trust_ceilings': len(default_ceilings),
            'propagation_rules': len(default_rules)
        }
    ).to_json())

@app.route('/api/v1/admin/reset', methods=['POST'])
@admin_required
def reset_system():
    """Reset the trust system (DANGEROUS - for development only)"""
    if app.config.get('ENV') == 'production':
        return jsonify(ApiResponse(
            success=False,
            error="Reset not allowed in production"
        ).to_json()), 403
    
    # Clear registry
    trust_authority.trust_registry.clear()
    trust_authority.propagation_rules.clear()
    trust_authority.save_registry()
    
    # Note: We don't clear the blockchain as it's immutable by design
    
    return jsonify(ApiResponse(
        success=True,
        data={'message': 'Trust registry reset (blockchain preserved)'}
    ).to_json())

# Error handlers
@app.errorhandler(400)
def bad_request(error):
    return jsonify(ApiResponse(
        success=False,
        error=str(error)
    ).to_json()), 400

@app.errorhandler(401)
def unauthorized(error):
    return jsonify(ApiResponse(
        success=False,
        error="Unauthorized"
    ).to_json()), 401

@app.errorhandler(403)
def forbidden(error):
    return jsonify(ApiResponse(
        success=False,
        error="Forbidden"
    ).to_json()), 403

@app.errorhandler(404)
def not_found(error):
    return jsonify(ApiResponse(
        success=False,
        error="Resource not found"
    ).to_json()), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify(ApiResponse(
        success=False,
        error="Internal server error"
    ).to_json()), 500

if __name__ == '__main__':
    # Development server
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=os.environ.get('ENV') != 'production'
    )
