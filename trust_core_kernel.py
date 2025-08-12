"""
AI Trust Chain Framework - Core Trust Kernel (Fixed with Tuple Materiality)
Copyright (C) 2025 Mossrake Group, LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This module implements the core trust and confidence propagation mechanics
with blockchain-based immutable audit trails stored on disk.

Based on the AI Trust Chain Framework designed by Mossrake Group, LLC.
The framework design and concepts are proprietary intellectual property
of Mossrake Group, LLC. This implementation is released under AGPL-3.0.

IMPORTANT: 
- Trust values are NEVER set by endpoints. They are calculated based on:
  * Trust ceilings (for root assertions)
  * Propagation rules (for derived assertions)
- Materiality is a tuple (weight, apply_to_trust) where:
  * weight: relative importance (0.0-1.0)
  * apply_to_trust: whether this input affects trust propagation
"""

import hashlib
import json
import time
import uuid
import warnings
from dataclasses import dataclass, field, asdict as _asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import sqlite3
import threading

# Thread-local storage for database connections
_thread_local = threading.local()

# Type alias for materiality specification
MaterialitySpec = Tuple[float, bool]  # (weight, apply_to_trust)

def asdict(obj):
    """Custom asdict that converts Enums to their values"""
    def dict_factory(field_list):
        result = {}
        for key, value in field_list:
            if isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result
    return _asdict(obj, dict_factory=dict_factory)

class EndpointType(Enum):
    """Enumeration of supported endpoint types"""
    SENSOR = "sensor"
    API = "api"
    ML_MODEL = "ml_model"
    LLM = "llm"
    PROCESSOR = "processor"
    DATA_SOURCE = "data_source"

@dataclass
class TrustMetadata:
    """
    Trust characterization metadata wrapper
    
    Trust is represented as a single composite value, with optional natural language
    explanation that may describe the various dimensions that contributed to the trust
    assessment (calibration status, environmental conditions, etc.)
    
    IMPORTANT: trust_value is ALWAYS calculated by the framework, never set directly
    """
    # Core values
    trust_value: float  # 0.0 to 1.0 - FRAMEWORK CALCULATED (never set by endpoints)
    confidence_value: float  # 0.0 to 1.0 - self-reported certainty from endpoint
    trust_explanation: Optional[str] = None  # Natural language explanation of trust factors
    
    # Common dimension - universally useful across all endpoint types
    temporal_validity: float = 1.0  # 0.0 to 1.0 - freshness/age factor (1.0 = fresh)
    
    # Metadata and tracking
    endpoint_id: str = ""
    endpoint_type: EndpointType = None
    endpoint_class: str = ""
    timestamp: float = field(default_factory=time.time)  # When assertion was created
    provenance: List[str] = field(default_factory=list)  # Chain of endpoint classes
    
    # Trust calculation details
    trust_input_count: int = 0  # Number of assertions that affected trust
    context_input_count: int = 0  # Number of assertions used for context only
    
    # Optional context
    limitations: Dict[str, Any] = field(default_factory=dict)  # Known limitations
    context: Dict[str, Any] = field(default_factory=dict)  # Additional context for consumers
    
    def validate(self) -> bool:
        """Validate all values are within bounds"""
        return (0.0 <= self.trust_value <= 1.0 and 
                0.0 <= self.confidence_value <= 1.0 and
                0.0 <= self.temporal_validity <= 1.0)

@dataclass
class Assertion:
    """Represents an assertion from an endpoint with dual-channel metadata"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    endpoint_id: str = ""
    content: Any = None
    metadata: TrustMetadata = None
    consumed_assertions: List[str] = field(default_factory=list)
    consumed_assertion_materiality: Dict[str, MaterialitySpec] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        """Convert assertion to dictionary for serialization"""
        return {
            'id': self.id,
            'endpoint_id': self.endpoint_id,
            'content': self.content,
            'metadata': asdict(self.metadata) if self.metadata else None,
            'consumed_assertions': self.consumed_assertions,
            'consumed_assertion_materiality': self.consumed_assertion_materiality,
            'timestamp': self.timestamp
        }

@dataclass
class Block:
    """Blockchain block for immutable audit trail"""
    index: int
    timestamp: float
    assertions: List[Dict]
    previous_hash: str
    nonce: int = 0
    hash: str = ""
    
    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of block contents"""
        block_string = json.dumps({
            'index': self.index,
            'timestamp': self.timestamp,
            'assertions': self.assertions,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def mine_block(self, difficulty: int = 2) -> None:
        """Simple proof-of-work mining"""
        target = '0' * difficulty
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()

@dataclass
class TrustRule:
    """Trust propagation rule definition"""
    rule_id: str
    endpoint_type: EndpointType
    max_trust_ceiling: float
    propagation_method: str  # 'minimum', 'weighted_average', 'consensus'
    consensus_threshold: Optional[float] = None
    weight_factors: Dict[str, float] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate rule parameters"""
        return 0.0 <= self.max_trust_ceiling <= 0.99

class TrustAuthority:
    """Centralized trust authority managing trust registry and propagation rules"""
    
    def __init__(self, registry_path: str = "trust_registry.json"):
        self.registry_path = Path(registry_path)
        self.trust_registry: Dict[str, float] = {}  # endpoint_class -> max_trust
        self.propagation_rules: Dict[str, TrustRule] = {}
        self.load_registry()
    
    def load_registry(self) -> None:
        """Load trust registry from disk"""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
                self.trust_registry = data.get('trust_registry', {})
                # Reconstruct TrustRule objects
                for rule_id, rule_data in data.get('propagation_rules', {}).items():
                    rule_data['endpoint_type'] = EndpointType(rule_data['endpoint_type'])
                    self.propagation_rules[rule_id] = TrustRule(**rule_data)
    
    def save_registry(self) -> None:
        """Persist trust registry to disk"""
        data = {
            'trust_registry': self.trust_registry,
            'propagation_rules': {
                rule_id: asdict(rule) for rule_id, rule in self.propagation_rules.items()
            }
        }
        # Note: asdict already converts EndpointType enums to strings, so no need to convert again
        
        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def set_endpoint_trust_ceiling(self, endpoint_class: str, max_trust: float) -> None:
        """Set maximum trust ceiling for an endpoint class"""
        if not 0.0 <= max_trust <= 1.0:
            raise ValueError("Trust value must be between 0.0 and 1.0")
        self.trust_registry[endpoint_class] = max_trust
        self.save_registry()
    
    def get_trust_ceiling(self, endpoint_class: str) -> float:
        """Get maximum trust ceiling for an endpoint class"""
        return self.trust_registry.get(endpoint_class, 1.0)  # Default to 1.0
    
    def add_propagation_rule(self, rule: TrustRule) -> None:
        """Add or update a trust propagation rule"""
        if not rule.validate():
            raise ValueError("Invalid trust rule parameters")
        self.propagation_rules[rule.rule_id] = rule
        self.save_registry()
    
    def get_rule_for_endpoint_type(self, endpoint_type: EndpointType) -> Optional[TrustRule]:
        """Get the propagation rule for a specific endpoint type"""
        for rule in self.propagation_rules.values():
            if rule.endpoint_type == endpoint_type:
                return rule
        return None
    
    def calculate_propagated_trust(self,
                                  consumed_trusts: List[Tuple[float, float]],
                                  endpoint_type: EndpointType,
                                  endpoint_class: str) -> float:
        """
        Calculate propagated trust value based on consumed assertions
        
        Args:
            consumed_trusts: List of (trust_value, weight) tuples 
                           ONLY for assertions where apply_to_trust=True
            endpoint_type: Type of the endpoint creating the assertion
            endpoint_class: Class of the endpoint for ceiling lookup
            
        Returns:
            Calculated trust value, capped by endpoint's trust ceiling
        """
        if not consumed_trusts:
            # No consumed assertions - return the trust ceiling for this endpoint class
            return self.get_trust_ceiling(endpoint_class)
        
        # Extract trust values and weights
        trust_values = [t for t, _ in consumed_trusts]
        weights = [w for _, w in consumed_trusts]
        
        # Normalize weights to sum to 1.0 if they don't
        total_weight = sum(weights)
        if total_weight > 0 and abs(total_weight - 1.0) > 0.01:
            weights = [w / total_weight for w in weights]
            consumed_trusts = [(t, w) for (t, _), w in zip(consumed_trusts, weights)]
        
        # Find applicable rule
        rule = self.get_rule_for_endpoint_type(endpoint_type)
        
        if not rule or rule.propagation_method == 'minimum':
            # Default to minimum propagation - most conservative
            trust = min(trust_values)
            
        elif rule.propagation_method == 'weighted_average':
            # Weighted average based on materiality weights
            if total_weight > 0:
                trust = sum(t * w for t, w in consumed_trusts)
            else:
                trust = min(trust_values)
                
        elif rule.propagation_method == 'consensus':
            # Consensus-based propagation
            threshold = rule.consensus_threshold or 0.7
            high_trust_values = [t for t in trust_values if t >= threshold]
            if len(high_trust_values) >= len(trust_values) * 0.5:
                # Majority have high trust - use their average
                trust = sum(high_trust_values) / len(high_trust_values)
            else:
                # No consensus - fall back to minimum
                trust = min(trust_values)
        else:
            # Unknown method - default to minimum
            trust = min(trust_values)
        
        # Apply trust ceiling constraint
        max_trust = self.get_trust_ceiling(endpoint_class)
        return min(trust, max_trust)

class BlockchainAuditTrail:
    """Blockchain-based immutable audit trail stored on disk using SQLite"""
    
    def __init__(self, db_path: str = "trust_chain.db"):
        self.db_path = db_path
        self.init_database()
        self.init_genesis_block()
    
    def get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection"""
        if not hasattr(_thread_local, 'connection'):
            _thread_local.connection = sqlite3.connect(self.db_path)
            _thread_local.connection.row_factory = sqlite3.Row
        return _thread_local.connection
    
    def init_database(self) -> None:
        """Initialize SQLite database schema"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS blocks (
                index_num INTEGER PRIMARY KEY,
                timestamp REAL NOT NULL,
                assertions TEXT NOT NULL,
                previous_hash TEXT NOT NULL,
                nonce INTEGER NOT NULL,
                hash TEXT NOT NULL UNIQUE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS assertions (
                id TEXT PRIMARY KEY,
                endpoint_id TEXT NOT NULL,
                content TEXT,
                metadata TEXT NOT NULL,
                consumed_assertions TEXT,
                consumed_assertion_materiality TEXT,
                timestamp REAL NOT NULL,
                block_hash TEXT,
                FOREIGN KEY (block_hash) REFERENCES blocks(hash)
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_assertions_endpoint 
            ON assertions(endpoint_id)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_assertions_timestamp 
            ON assertions(timestamp)
        ''')
        
        conn.commit()
    
    def init_genesis_block(self) -> None:
        """Create genesis block if chain is empty"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) as count FROM blocks")
        if cursor.fetchone()['count'] == 0:
            genesis = Block(
                index=0,
                timestamp=time.time(),
                assertions=[],
                previous_hash="0"
            )
            genesis.mine_block()
            self._save_block(genesis)
    
    def _save_block(self, block: Block) -> None:
        """Save block to database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO blocks (index_num, timestamp, assertions, previous_hash, nonce, hash)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            block.index,
            block.timestamp,
            json.dumps(block.assertions),
            block.previous_hash,
            block.nonce,
            block.hash
        ))
        
        # Link assertions to block
        for assertion_dict in block.assertions:
            cursor.execute('''
                UPDATE assertions SET block_hash = ? WHERE id = ?
            ''', (block.hash, assertion_dict['id']))
        
        conn.commit()
    
    def get_latest_block(self) -> Optional[Block]:
        """Retrieve the latest block from the chain"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM blocks ORDER BY index_num DESC LIMIT 1
        ''')
        
        row = cursor.fetchone()
        if row:
            return Block(
                index=row['index_num'],
                timestamp=row['timestamp'],
                assertions=json.loads(row['assertions']),
                previous_hash=row['previous_hash'],
                nonce=row['nonce'],
                hash=row['hash']
            )
        return None
    
    def add_assertion(self, assertion: Assertion) -> str:
        """Add assertion to pending pool and database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        assertion_dict = assertion.to_dict()
        
        # Convert metadata to dict and handle enum serialization
        if assertion.metadata:
            metadata_dict = asdict(assertion.metadata)
            metadata_json = json.dumps(metadata_dict)
        else:
            metadata_json = '{}'
        
        # Serialize materiality specs
        materiality_json = json.dumps(assertion.consumed_assertion_materiality)
        
        cursor.execute('''
            INSERT INTO assertions (id, endpoint_id, content, metadata, consumed_assertions, 
                                  consumed_assertion_materiality, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            assertion.id,
            assertion.endpoint_id,
            json.dumps(assertion.content),
            metadata_json,
            json.dumps(assertion.consumed_assertions),
            materiality_json,
            assertion.timestamp
        ))
        
        conn.commit()
        return assertion.id
    
    def create_block(self, assertions: List[Assertion]) -> Block:
        """Create and mine a new block with assertions"""
        latest_block = self.get_latest_block()
        
        new_block = Block(
            index=latest_block.index + 1 if latest_block else 1,
            timestamp=time.time(),
            assertions=[a.to_dict() for a in assertions],
            previous_hash=latest_block.hash if latest_block else "0"
        )
        
        new_block.mine_block()
        self._save_block(new_block)
        
        return new_block
    
    def get_assertion_trail(self, assertion_id: str) -> List[Dict]:
        """Retrieve complete audit trail for an assertion"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        trail = []
        to_process = [assertion_id]
        processed = set()
        
        while to_process:
            current_id = to_process.pop(0)
            if current_id in processed:
                continue
            
            cursor.execute('''
                SELECT * FROM assertions WHERE id = ?
            ''', (current_id,))
            
            row = cursor.fetchone()
            if row:
                assertion_data = {
                    'id': row['id'],
                    'endpoint_id': row['endpoint_id'],
                    'content': json.loads(row['content']) if row['content'] else None,
                    'metadata': json.loads(row['metadata']),
                    'consumed_assertions': json.loads(row['consumed_assertions']),
                    'consumed_assertion_materiality': json.loads(row['consumed_assertion_materiality'] or '{}'),
                    'timestamp': row['timestamp'],
                    'block_hash': row['block_hash']
                }
                trail.append(assertion_data)
                
                # Add consumed assertions to processing queue
                to_process.extend(assertion_data['consumed_assertions'])
            
            processed.add(current_id)
        
        return trail
    
    def verify_chain_integrity(self) -> bool:
        """Verify the integrity of the entire blockchain"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM blocks ORDER BY index_num
        ''')
        
        previous_hash = "0"
        for row in cursor.fetchall():
            block = Block(
                index=row['index_num'],
                timestamp=row['timestamp'],
                assertions=json.loads(row['assertions']),
                previous_hash=row['previous_hash'],
                nonce=row['nonce'],
                hash=row['hash']
            )
            
            # Verify hash
            if block.hash != block.calculate_hash():
                return False
            
            # Verify chain linkage
            if block.previous_hash != previous_hash:
                return False
            
            previous_hash = block.hash
        
        return True

class TrustKernel:
    """Main trust kernel coordinating trust evaluation and propagation"""
    
    def __init__(self, 
                 authority: TrustAuthority,
                 audit_trail: BlockchainAuditTrail):
        self.authority = authority
        self.audit_trail = audit_trail
        self.pending_assertions: List[Assertion] = []
        self.assertion_cache: Dict[str, Assertion] = {}
    
    def create_assertion(self,
                        endpoint_id: str,
                        endpoint_class: str,
                        endpoint_type: EndpointType,
                        content: Any,
                        confidence: float,
                        trust_explanation: str,  # REQUIRED - endpoints must explain their trust
                        consumed_assertion_ids: Optional[List[str]] = None,
                        consumed_assertion_materiality: Optional[Dict[str, MaterialitySpec]] = None,
                        limitations: Optional[Dict[str, Any]] = None,
                        temporal_validity: Optional[float] = None) -> Assertion:
        """
        Create a new assertion with trust evaluation
        
        Args:
            endpoint_id: Unique identifier for the endpoint
            endpoint_class: Class/type for trust ceiling lookup (e.g., "sensor.temp.honeywell")
            endpoint_type: Type of endpoint (sensor, ml_model, llm, etc.)
            content: The actual assertion content/data
            confidence: Self-reported confidence (0.0-1.0)
            trust_explanation: REQUIRED explanation of trust status from the endpoint
                - For root assertions: why the endpoint believes it's trustworthy
                - For derived assertions: interpretation of consumed assertions' trust
            consumed_assertion_ids: List of assertion IDs this assertion is based on
            consumed_assertion_materiality: Dict mapping assertion_id to (weight, apply_to_trust) tuple
                - weight: 0.0-1.0 relative importance of this input
                - apply_to_trust: whether this input should affect trust propagation
            limitations: Any limitations or caveats about this assertion
            temporal_validity: Optional temporal validity (defaults to 1.0)
        
        Example:
            consumed_assertion_materiality={
                "sensor-123": (0.7, True),   # 70% weight, affects trust
                "sensor-456": (0.3, True),   # 30% weight, affects trust
                "debug-789": (0.0, False),   # Context only, no trust impact
            }
        
        Note: 
        - Trust value is ALWAYS calculated by the framework, never passed as parameter
        - Trust explanation is ALWAYS provided by the endpoint, never auto-generated
        - For root assertions (no consumed assertions): trust = trust ceiling
        - For derived assertions: trust = propagated trust, capped by ceiling
        - Only assertions with apply_to_trust=True affect trust propagation
        """
        consumed_assertion_ids = consumed_assertion_ids or []
        consumed_assertion_materiality = consumed_assertion_materiality or {}
        limitations = limitations or {}
        provenance_chain = []
        
        # Validate confidence
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")
        
        # Validate required trust_explanation
        # endpoint MAY use an AI to assess consumed assertions to develop the trust_explanation
        if not trust_explanation:
            raise ValueError("trust_explanation is required - endpoints must explain their trust status")
        
        # Calculate trust based on consumed assertions
        consumed_trusts = []  # Only includes assertions where apply_to_trust=True
        trust_input_count = 0
        context_input_count = 0
        
        if consumed_assertion_ids:
            # Validate and process materiality specs
            trust_weights = []
            
            for assertion_id in consumed_assertion_ids:
                consumed = self.get_assertion(assertion_id)
                if consumed and consumed.metadata:
                    # Get materiality spec with defaults
                    mat_spec = consumed_assertion_materiality.get(assertion_id, (1.0, True))
                    
                    # Validate tuple format
                    if not isinstance(mat_spec, tuple) or len(mat_spec) != 2:
                        raise ValueError(f"Materiality spec for {assertion_id} must be (weight, apply_to_trust) tuple")
                    
                    weight, apply_to_trust = mat_spec
                    
                    # Validate weight
                    if not 0.0 <= weight <= 1.0:
                        warnings.warn(f"Weight for {assertion_id} should be 0.0-1.0, got {weight}")
                        weight = max(0.0, min(1.0, weight))  # Clamp to valid range
                    
                    if apply_to_trust:
                        # This assertion affects trust propagation
                        consumed_trusts.append((consumed.metadata.trust_value, weight))
                        trust_weights.append(weight)
                        trust_input_count += 1
                    else:
                        # This assertion is for context only
                        context_input_count += 1
                    
                    # Build provenance chain from all consumed assertions
                    if consumed.metadata.provenance:
                        for endpoint in consumed.metadata.provenance:
                            if endpoint not in provenance_chain:
                                provenance_chain.append(endpoint)
            
            # Validate that trust weights sum to approximately 1.0
            if trust_weights:
                total_weight = sum(trust_weights)
                if abs(total_weight - 1.0) > 0.1:
                    warnings.warn(
                        f"Trust-affecting weights sum to {total_weight:.2f}, expected ~1.0. "
                        f"Weights will be normalized."
                    )
        
        # Add current endpoint to the chain
        if endpoint_class not in provenance_chain:
            provenance_chain.append(endpoint_class)
        
        # Calculate trust value
        if consumed_trusts:
            # Derived assertion: use propagation rules
            trust_value = self.authority.calculate_propagated_trust(
                consumed_trusts, endpoint_type, endpoint_class
            )
        else:
            # Root assertion: trust equals the ceiling for this endpoint class
            trust_value = self.authority.get_trust_ceiling(endpoint_class)
        
        # Set temporal validity
        if temporal_validity is None:
            temporal_validity = 1.0
        elif not 0.0 <= temporal_validity <= 1.0:
            warnings.warn(f"Temporal validity should be 0.0-1.0, got {temporal_validity}")
            temporal_validity = max(0.0, min(1.0, temporal_validity))
        
        # Create metadata
        metadata = TrustMetadata(
            trust_value=trust_value,  # CALCULATED, not provided
            confidence_value=confidence,
            trust_explanation=trust_explanation,
            temporal_validity=temporal_validity,
            endpoint_id=endpoint_id,
            endpoint_type=endpoint_type,
            endpoint_class=endpoint_class,
            timestamp=time.time(),
            provenance=provenance_chain,
            trust_input_count=trust_input_count,
            context_input_count=context_input_count,
            limitations=limitations
        )
        
        # Create assertion
        assertion = Assertion(
            endpoint_id=endpoint_id,
            content=content,
            metadata=metadata,
            consumed_assertions=consumed_assertion_ids,
            consumed_assertion_materiality=consumed_assertion_materiality
        )
        
        # Add to audit trail
        assertion_id = self.audit_trail.add_assertion(assertion)
        assertion.id = assertion_id
        
        # Cache and add to pending
        self.assertion_cache[assertion_id] = assertion
        self.pending_assertions.append(assertion)
        
        # Create block if enough pending assertions
        if len(self.pending_assertions) >= 10:
            self.commit_pending_assertions()
        
        return assertion
    
    def get_assertion(self, assertion_id: str) -> Optional[Assertion]:
        """Retrieve an assertion by ID"""
        if assertion_id in self.assertion_cache:
            return self.assertion_cache[assertion_id]
        
        # Query from audit trail
        trail = self.audit_trail.get_assertion_trail(assertion_id)
        if trail:
            assertion_data = trail[0]
            metadata_dict = assertion_data['metadata']
            metadata = TrustMetadata(
                trust_value=metadata_dict['trust_value'],
                confidence_value=metadata_dict['confidence_value'],
                trust_explanation=metadata_dict.get('trust_explanation'),
                temporal_validity=metadata_dict.get('temporal_validity', 1.0),
                endpoint_id=metadata_dict['endpoint_id'],
                endpoint_type=EndpointType(metadata_dict['endpoint_type']),
                endpoint_class=metadata_dict.get('endpoint_class', ''),
                timestamp=metadata_dict['timestamp'],
                provenance=metadata_dict.get('provenance', []),
                trust_input_count=metadata_dict.get('trust_input_count', 0),
                context_input_count=metadata_dict.get('context_input_count', 0),
                limitations=metadata_dict.get('limitations', {}),
                context=metadata_dict.get('context', {})
            )
            
            # Reconstruct materiality specs
            mat_dict = assertion_data.get('consumed_assertion_materiality', {})
            materiality_specs = {}
            for aid, spec in mat_dict.items():
                if isinstance(spec, list) and len(spec) == 2:
                    materiality_specs[aid] = tuple(spec)
                else:
                    materiality_specs[aid] = (1.0, True)  # Default
            
            assertion = Assertion(
                id=assertion_data['id'],
                endpoint_id=assertion_data['endpoint_id'],
                content=assertion_data['content'],
                metadata=metadata,
                consumed_assertions=assertion_data['consumed_assertions'],
                consumed_assertion_materiality=materiality_specs,
                timestamp=assertion_data['timestamp']
            )
            
            self.assertion_cache[assertion_id] = assertion
            return assertion
        
        return None
    
    def commit_pending_assertions(self) -> Optional[Block]:
        """Commit pending assertions to a new block"""
        if not self.pending_assertions:
            return None
        
        block = self.audit_trail.create_block(self.pending_assertions)
        self.pending_assertions.clear()
        
        return block
    
    def get_trust_confidence_matrix(self, assertion_id: str) -> Dict[str, Any]:
        """
        Get trust-confidence matrix classification for an assertion
        
        This is computed on-demand when decision support is needed,
        not stored with each assertion.
        """
        assertion = self.get_assertion(assertion_id)
        if not assertion or not assertion.metadata:
            return {"status": "unknown"}
        
        trust = assertion.metadata.trust_value
        confidence = assertion.metadata.confidence_value
        
        if trust >= 0.7 and confidence >= 0.7:
            return {
                "status": "high_trust_high_confidence",
                "recommendation": "Proceed with minimal oversight",
                "trust_value": trust,
                "confidence_value": confidence
            }
        elif trust >= 0.7 and confidence < 0.7:
            return {
                "status": "high_trust_low_confidence",
                "recommendation": "Apply human judgment to ambiguous situations",
                "trust_value": trust,
                "confidence_value": confidence
            }
        elif trust < 0.7 and confidence >= 0.7:
            return {
                "status": "low_trust_high_confidence",
                "recommendation": "Exercise extreme caution due to potential overconfidence",
                "trust_value": trust,
                "confidence_value": confidence
            }
        else:
            return {
                "status": "low_trust_low_confidence",
                "recommendation": "Seek alternative information sources or defer decisions",
                "trust_value": trust,
                "confidence_value": confidence
            }
    
    def get_assertion_provenance(self, assertion_id: str) -> Dict:
        """Get complete provenance chain for an assertion"""
        trail = self.audit_trail.get_assertion_trail(assertion_id)
        
        return {
            "assertion_id": assertion_id,
            "trail_length": len(trail),
            "chain": trail,
            "integrity_verified": self.audit_trail.verify_chain_integrity()
        }
    
    def show_evidence_chain(self, assertion_id: str) -> Dict:
        """
        'Show me the evidence' - Get complete reasoning chain for an assertion
        
        Returns a structured tree showing how this assertion was derived,
        including all consumed assertions, their trust/confidence values,
        explanations, and materiality specifications.
        """
        def build_assertion_tree(aid: str, visited: set = None) -> Dict:
            """Recursively build the assertion tree"""
            if visited is None:
                visited = set()
            
            if aid in visited:
                return {"circular_reference": aid}
            
            visited.add(aid)
            
            assertion = self.get_assertion(aid)
            if not assertion:
                return {"error": f"Assertion {aid} not found"}
            
            # Build node information
            node = {
                "assertion_id": aid,
                "endpoint_id": assertion.endpoint_id,
                "endpoint_type": assertion.metadata.endpoint_type.value if assertion.metadata else "unknown",
                "endpoint_class": assertion.metadata.endpoint_class if assertion.metadata else "unknown",
                "content": assertion.content,
                "trust_value": assertion.metadata.trust_value if assertion.metadata else None,
                "confidence_value": assertion.metadata.confidence_value if assertion.metadata else None,
                "temporal_validity": assertion.metadata.temporal_validity if assertion.metadata else None,
                "trust_explanation": assertion.metadata.trust_explanation if assertion.metadata else None,
                "trust_input_count": assertion.metadata.trust_input_count if assertion.metadata else 0,
                "context_input_count": assertion.metadata.context_input_count if assertion.metadata else 0,
                "timestamp": assertion.timestamp,
                "consumed_assertions": []
            }
            
            # Recursively build tree for consumed assertions
            for consumed_id in assertion.consumed_assertions:
                child_node = build_assertion_tree(consumed_id, visited.copy())
                # Add materiality info
                mat_spec = assertion.consumed_assertion_materiality.get(consumed_id, (1.0, True))
                child_node["materiality_weight"] = mat_spec[0]
                child_node["affects_trust"] = mat_spec[1]
                node["consumed_assertions"].append(child_node)
            
            return node
        
        # Build the tree
        tree = build_assertion_tree(assertion_id)
        
        # Generate summary
        summary = self._generate_evidence_summary(tree)
        
        return {
            "query_assertion_id": assertion_id,
            "evidence_tree": tree,
            "summary": summary,
            "timestamp": time.time()
        }
    
    def _generate_evidence_summary(self, tree: Dict) -> Dict:
        """Generate a summary of the evidence chain"""
        
        def count_nodes(node: Dict) -> tuple:
            """Count total nodes and depth"""
            if "error" in node or "circular_reference" in node:
                return 0, 0
            
            if not node.get("consumed_assertions"):
                return 1, 1
            
            child_counts = [count_nodes(child) for child in node["consumed_assertions"]]
            total_nodes = 1 + sum(c[0] for c in child_counts)
            max_depth = 1 + max(c[1] for c in child_counts)
            
            return total_nodes, max_depth
        
        def find_weakest_link(node: Dict, path: str = "") -> tuple:
            """Find the assertion with lowest trust in the chain"""
            if "error" in node or "circular_reference" in node:
                return None, None, ""
            
            current_path = f"{path}/{node['endpoint_id']}"
            weakest = (node["trust_value"], node, current_path)
            
            for child in node.get("consumed_assertions", []):
                # Only consider assertions that affect trust
                if child.get("affects_trust", True):
                    child_weakest = find_weakest_link(child, current_path)
                    if child_weakest[0] is not None and (weakest[0] is None or child_weakest[0] < weakest[0]):
                        weakest = child_weakest
            
            return weakest
        
        total_nodes, max_depth = count_nodes(tree)
        trust_value, weakest_node, weakest_path = find_weakest_link(tree)
        
        # Build trust confidence matrix for the root
        matrix = self.get_trust_confidence_matrix(tree["assertion_id"])
        
        summary = {
            "total_assertions": total_nodes,
            "chain_depth": max_depth,
            "root_trust": tree.get("trust_value"),
            "root_confidence": tree.get("confidence_value"),
            "trust_inputs": tree.get("trust_input_count", 0),
            "context_inputs": tree.get("context_input_count", 0),
            "weakest_link": {
                "trust_value": trust_value,
                "endpoint_id": weakest_node["endpoint_id"] if weakest_node else None,
                "path": weakest_path,
                "explanation": weakest_node.get("trust_explanation") if weakest_node else None
            } if weakest_node else None,
            "trust_confidence_assessment": matrix,
            "recommendation": matrix.get("recommendation")
        }
        
        return summary
    
    def explain_assertion_chain(self, assertion_id: str, format: str = "text") -> str:
        """
        Generate human-readable explanation of the assertion chain
        
        Args:
            assertion_id: The assertion to explain
            format: Output format - "text" for plain text, "markdown" for formatted
        
        Returns:
            Human-readable explanation of the evidence chain
        """
        evidence = self.show_evidence_chain(assertion_id)
        
        if format == "markdown":
            return self._format_evidence_markdown(evidence)
        else:
            return self._format_evidence_text(evidence)
    
    def _format_evidence_text(self, evidence: Dict) -> str:
        """Format evidence chain as plain text"""
        lines = []
        
        lines.append("EVIDENCE CHAIN ANALYSIS")
        lines.append("=" * 50)
        
        summary = evidence["summary"]
        lines.append(f"\nChain Depth: {summary['chain_depth']} levels")
        lines.append(f"Total Assertions: {summary['total_assertions']}")
        lines.append(f"Trust-Affecting Inputs: {summary['trust_inputs']}")
        lines.append(f"Context-Only Inputs: {summary['context_inputs']}")
        lines.append(f"\nFinal Trust: {summary['root_trust']:.2f}")
        lines.append(f"Final Confidence: {summary['root_confidence']:.2f}")
        lines.append(f"Assessment: {summary['trust_confidence_assessment']['status']}")
        lines.append(f"Recommendation: {summary['recommendation']}")
        
        if summary.get('weakest_link'):
            lines.append(f"\nWeakest Link: {summary['weakest_link']['endpoint_id']}")
            lines.append(f"  Trust: {summary['weakest_link']['trust_value']:.2f}")
            if summary['weakest_link'].get('explanation'):
                lines.append(f"  Issue: {summary['weakest_link']['explanation']}")
        
        lines.append("\nDETAILED CHAIN:")
        lines.append("-" * 50)
        
        def format_node(node: Dict, indent: int = 0) -> None:
            """Recursively format each node"""
            if "error" in node or "circular_reference" in node:
                return
            
            prefix = "  " * indent + "└─ " if indent > 0 else ""
            affects_trust = node.get("affects_trust", True)
            weight = node.get("materiality_weight", 1.0)
            
            trust_indicator = "✓" if affects_trust else "○"
            lines.append(f"{prefix}{trust_indicator} {node['endpoint_id']} ({node['endpoint_type']})")
            
            if indent > 0:  # Show materiality for consumed assertions
                lines.append(f"{'  ' * (indent + 1)}Weight: {weight:.2f}, "
                           f"Affects Trust: {affects_trust}")
            
            lines.append(f"{'  ' * (indent + 1)}Trust: {node['trust_value']:.2f}, "
                        f"Confidence: {node['confidence_value']:.2f}, "
                        f"Temporal: {node['temporal_validity']:.2f}")
            
            if node.get('trust_explanation'):
                lines.append(f"{'  ' * (indent + 1)}Explanation: {node['trust_explanation']}")
            
            if node.get('content'):
                lines.append(f"{'  ' * (indent + 1)}Content: {node['content']}")
            
            for child in node.get('consumed_assertions', []):
                format_node(child, indent + 1)
        
        format_node(evidence['evidence_tree'])
        
        return "\n".join(lines)
    
    def _format_evidence_markdown(self, evidence: Dict) -> str:
        """Format evidence chain as markdown"""
        lines = []
        
        lines.append("# Evidence Chain Analysis\n")
        
        summary = evidence["summary"]
        lines.append("## Summary\n")
        lines.append(f"- **Chain Depth**: {summary['chain_depth']} levels")
        lines.append(f"- **Total Assertions**: {summary['total_assertions']}")
        lines.append(f"- **Trust-Affecting Inputs**: {summary['trust_inputs']}")
        lines.append(f"- **Context-Only Inputs**: {summary['context_inputs']}")
        lines.append(f"- **Final Trust**: {summary['root_trust']:.2f}")
        lines.append(f"- **Final Confidence**: {summary['root_confidence']:.2f}")
        lines.append(f"- **Assessment**: `{summary['trust_confidence_assessment']['status']}`")
        lines.append(f"- **Recommendation**: {summary['recommendation']}\n")
        
        if summary.get('weakest_link'):
            lines.append("### ⚠️ Weakest Link\n")
            lines.append(f"- **Endpoint**: `{summary['weakest_link']['endpoint_id']}`")
            lines.append(f"- **Trust**: {summary['weakest_link']['trust_value']:.2f}")
            if summary['weakest_link'].get('explanation'):
                lines.append(f"- **Issue**: {summary['weakest_link']['explanation']}\n")
        
        lines.append("## Detailed Evidence Chain\n")
        
        def format_node(node: Dict, level: int = 0) -> None:
            """Recursively format each node"""
            if "error" in node or "circular_reference" in node:
                return
            
            indent = "  " * level
            bullet = "- " if level == 0 else "  - "
            affects_trust = node.get("affects_trust", True)
            weight = node.get("materiality_weight", 1.0)
            
            trust_indicator = "✅" if affects_trust else "📎"
            lines.append(f"{indent}{bullet}{trust_indicator} **{node['endpoint_id']}** (`{node['endpoint_type']}`)")
            
            if level > 0:  # Show materiality for consumed assertions
                lines.append(f"{indent}  - Materiality: {weight:.1%} | Trust Impact: {affects_trust}")
            
            lines.append(f"{indent}  - Trust: {node['trust_value']:.2f} | "
                        f"Confidence: {node['confidence_value']:.2f} | "
                        f"Temporal: {node['temporal_validity']:.2f}")
            
            if node.get('trust_explanation'):
                lines.append(f"{indent}  - 💭 {node['trust_explanation']}")
            
            if node.get('content'):
                lines.append(f"{indent}  - 📊 `{node['content']}`")
            
            for child in node.get('consumed_assertions', []):
                format_node(child, level + 1)
        
        format_node(evidence['evidence_tree'])
        
        return "\n".join(lines)
