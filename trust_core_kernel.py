"""
AI Trust Chain Framework - Core Trust Kernel
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
"""

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field, asdict as _asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import sqlite3
import threading

# Thread-local storage for database connections
_thread_local = threading.local()

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

@dataclass
class TrustMetadata:
    """
    Trust characterization metadata wrapper
    
    Trust is represented as a single composite value, with optional natural language
    explanation that may describe the various dimensions that contributed to the trust
    assessment (calibration status, environmental conditions, etc.)
    """
    # Core values
    trust_value: float  # 0.0 to 1.0 - composite trust value (incorporates all factors)
    confidence_value: float  # 0.0 to 1.0 - self-reported certainty
    trust_explanation: Optional[str] = None  # Natural language explanation of trust factors
    
    # Common dimension - universally useful across all endpoint types
    temporal_validity: float = 1.0  # 0.0 to 1.0 - freshness/age factor (1.0 = fresh)
    
    # Metadata and tracking
    endpoint_id: str = ""
    endpoint_type: EndpointType = None
    endpoint_class: str = ""
    timestamp: float = field(default_factory=time.time)  # When assertion was created
    provenance: List[str] = field(default_factory=list)  # Chain of endpoint classes
    
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
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        """Convert assertion to dictionary for serialization"""
        return {
            'id': self.id,
            'endpoint_id': self.endpoint_id,
            'content': self.content,
            'metadata': asdict(self.metadata) if self.metadata else None,
            'consumed_assertions': self.consumed_assertions,
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
        # Convert EndpointType enums to strings for JSON serialization
        for rule_data in data['propagation_rules'].values():
            rule_data['endpoint_type'] = rule_data['endpoint_type'].value
        
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
    
    def calculate_propagated_trust(self,
                                  consumed_trusts: List[Tuple[float, float]],
                                  endpoint_type: EndpointType,
                                  endpoint_class: str) -> float:
        """
        Calculate propagated trust value based on consumed assertions
        consumed_trusts: List of (trust_value, weight) tuples
        """
        if not consumed_trusts:
            return self.get_trust_ceiling(endpoint_class)
        
        # Extract trust values and weights
        trust_values = [t for t, _ in consumed_trusts]
        weights = [w for _, w in consumed_trusts]
        
        # Find applicable rule
        rule = None
        for r in self.propagation_rules.values():
            if r.endpoint_type == endpoint_type:
                rule = r
                break
        
        if not rule:
            # Default to minimum propagation if no rule defined
            trust = min(trust_values)
        else:
            if rule.propagation_method == 'minimum':
                trust = min(trust_values)
            elif rule.propagation_method == 'weighted_average':
                total_weight = sum(weights)
                if total_weight > 0:
                    trust = sum(t * w for t, w in consumed_trusts) / total_weight
                else:
                    trust = min(trust_values)
            elif rule.propagation_method == 'consensus':
                threshold = rule.consensus_threshold or 0.7
                high_trust_values = [t for t in trust_values if t >= threshold]
                if len(high_trust_values) >= len(trust_values) * 0.5:
                    trust = sum(high_trust_values) / len(high_trust_values)
                else:
                    trust = min(trust_values)
            else:
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
            # Convert EndpointType enum to string for JSON serialization
            metadata_dict = asdict(assertion.metadata)
            metadata_json = json.dumps(metadata_dict)
        else:
            metadata_json = '{}'
        
        cursor.execute('''
            INSERT INTO assertions (id, endpoint_id, content, metadata, consumed_assertions, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            assertion.id,
            assertion.endpoint_id,
            json.dumps(assertion.content),
            metadata_json,
            json.dumps(assertion.consumed_assertions),
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
                        trust_value: float = 0.,
                        trust_explanation = None,
                        consumed_assertion_ids: List[str] = None,
                        limitations: Dict[str, Any] = None) -> Assertion:
        """
        Create a new assertion with trust evaluation
        
        Args:
            endpoint_id: Unique identifier for the endpoint
            endpoint_class: Class/type for trust ceiling lookup (e.g., "sensor.temp.honeywell")
            endpoint_type: Type of endpoint (sensor, ml_model, llm, etc.)
            content: The actual assertion content/data
            confidence: Self-reported confidence (0.0-1.0)
            trust_value: Optional trust value - ONLY used for root assertions without consumed assertions
            consumed_assertion_ids: List of assertion IDs this assertion is based on
            limitations: Any limitations or caveats about this assertion
        
        Note: 
        
        Sophisticated endpoints may use an AI (SLM or LLM) or other logic to INTERPRET trust 
        explanations from consumed assertions and adjust trust accordingly. 
        
        An AI is NOT required to GENERATE assertions / trust explanations.

        """
        consumed_assertion_ids = consumed_assertion_ids or []
        limitations = limitations or {}
        provenance_chain = []

        # Calculate trust based on consumed assertions
        # Note: Materiality is determined internally by the endpoint
        # This reference implementation uses equal weights for simplicity
        consumed_trusts = []

        for assertion_id in consumed_assertion_ids:
            consumed = self.get_assertion(assertion_id)
            if consumed and consumed.metadata:
                # In a real endpoint, materiality would be determined based on
                # the endpoint's understanding of input importance
                # Example:
                # materiality = self.determine_materiality(consumed, self.context)
                materiality = 1.0  # Equal weight for reference implementation
                consumed_trusts.append((consumed.metadata.trust_value, materiality))
                
                # Build provenance chain
                if consumed.metadata.provenance:
                    # Merge provenance chains from consumed assertions
                    for endpoint in consumed.metadata.provenance:
                        if endpoint not in provenance_chain:
                            provenance_chain.append(endpoint)

                # STUB: Sophisticated endpoints with AI (LLM or SLM) access can INTERPRET explanations
                # from consumed assertions to make trust and materiality adjustments:
                # 
                # if consumed.metadata.trust_explanation:
                #     # Use AI to understand if "sensor uncalibrated for 3 years" matters
                #     understanding = self.understand_trust_explanation(consumed.metadata.trust_explanation, "ml_model")
                #     if understanding:
                #         adjusted_trust = ...
                #         # Override trust if context makes it more/less relevant
                #         consumed_trusts[-1] = (adjusted_trust, materiality)

        # Add current endpoint to the chain
        if endpoint_class not in provenance_chain:
            provenance_chain.append(endpoint_class)
       
        # Calculate propagated trust
        if consumed_trusts:
            # When there are consumed assertions, use propagation rules
            trust_value = self.authority.calculate_propagated_trust(
                consumed_trusts, endpoint_type, endpoint_class
            )
        elif trust_value == 0.:
            # No consumed assertions and no trust provided, use ceiling
            trust_value = self.authority.get_trust_ceiling(endpoint_class)
        
        # Generate trust explanation as appropriate
        if trust_explanation is None:
            trust_explanation = 'ok'
        
        # STUB: Endpoints generate explanations
        # Simple example:
        # if calculated_trust < 0.5:
        #     trust_explanation = f"Trust is low ({calculated_trust:.2f}) due to upstream data quality issues"
        
        metadata = TrustMetadata(
            trust_value=trust_value,
            confidence_value=confidence,
            trust_explanation=trust_explanation,
            temporal_validity=1.0,  # Could be calculated based on data age
            endpoint_id=endpoint_id,
            endpoint_type=endpoint_type,
            endpoint_class=endpoint_class,
            timestamp=time.time(),
            provenance=provenance_chain,
            limitations=limitations
        )
        
        # Create assertion
        assertion = Assertion(
            endpoint_id=endpoint_id,
            content=content,
            metadata=metadata,
            consumed_assertions=consumed_assertion_ids
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
                limitations=metadata_dict.get('limitations', {}),
                context=metadata_dict.get('context', {})
            )
            
            assertion = Assertion(
                id=assertion_data['id'],
                endpoint_id=assertion_data['endpoint_id'],
                content=assertion_data['content'],
                metadata=metadata,
                consumed_assertions=assertion_data['consumed_assertions'],
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
    
    def get_trust_confidence_matrix(self, assertion_id: str) -> Dict[str, str]:
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
        and explanations at each step.
        
        This is the key function for business leaders to understand the
        complete evidence chain behind an AI recommendation.
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
                "timestamp": assertion.timestamp,
                "consumed_assertions": []
            }
            
            # Recursively build tree for consumed assertions
            for consumed_id in assertion.consumed_assertions:
                child_node = build_assertion_tree(consumed_id, visited.copy())
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
            lines.append(f"{prefix}{node['endpoint_id']} ({node['endpoint_type']})")
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
            
            lines.append(f"{indent}{bullet}**{node['endpoint_id']}** (`{node['endpoint_type']}`)")
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
