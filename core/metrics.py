"""
Metrics tracking and monitoring
"""

from typing import Dict, Any, List
from datetime import datetime
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)


class MetricsTracker:
    """Track performance metrics for the RAG system"""
    
    def __init__(self, log_file: str = "metrics.jsonl"):
        self.log_file = Path(log_file)
        self.current_session = []
    
    def log_query(self, query: str, query_profile: Dict[str, Any], 
                  strategy_used: str, response_time: float, 
                  result: Dict[str, Any]):
        """Log a query execution"""
        
        metric = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'query_type': query_profile.get('query_type'),
            'complexity': query_profile.get('complexity'),
            'strategy': strategy_used,
            'response_time': response_time,
            'num_sources': result.get('num_sources', 0),
            'confidence': result.get('confidence', None)
        }
        
        self.current_session.append(metric)
        
        # Append to log file
        self._append_to_log(metric)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics for current session"""
        
        if not self.current_session:
            return {}
        
        total_queries = len(self.current_session)
        avg_response_time = sum(m['response_time'] for m in self.current_session) / total_queries
        
        # Count by strategy
        strategy_counts = {}
        for metric in self.current_session:
            strategy = metric['strategy']
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        # Count by query type
        type_counts = {}
        for metric in self.current_session:
            qtype = metric['query_type']
            type_counts[qtype] = type_counts.get(qtype, 0) + 1
        
        return {
            'total_queries': total_queries,
            'avg_response_time': avg_response_time,
            'strategy_distribution': strategy_counts,
            'query_type_distribution': type_counts
        }
    
    def _append_to_log(self, metric: Dict[str, Any]):
        """Append metric to log file"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(metric) + '\n')
        except Exception as e:
            logger.warning(f"Failed to write metric to log: {e}")
