"""
Output formatting utilities
"""

from typing import List, Dict, Any, Optional


def format_sources(sources: List[Dict[str, Any]], max_preview_length: int = 200) -> str:
    """
    Format source information for display
    
    Args:
        sources: List of source dictionaries
        max_preview_length: Maximum length for preview text
        
    Returns:
        Formatted string
    """
    if not sources:
        return "No sources available."
    
    lines = [f"\nSources ({len(sources)}):"]
    for source in sources:
        rank = source.get("rank", "?")
        similarity = source.get("similarity_score", 0.0)
        distance = source.get("distance", 0.0)
        source_name = source.get("source", "unknown")
        preview = source.get("preview", "")
        
        if len(preview) > max_preview_length:
            preview = preview[:max_preview_length] + "..."
        
        lines.append(f"  [{rank}] Similarity: {similarity:.3f} | Distance: {distance:.3f}")
        lines.append(f"      Source: {source_name}")
        if preview:
            lines.append(f"      Preview: {preview}\n")
    
    return "\n".join(lines)


def format_results(result: Dict[str, Any], include_context: bool = False) -> str:
    """
    Format RAG query results for display
    
    Args:
        result: Result dictionary from RAG pipeline
        include_context: Whether to include context in output
        
    Returns:
        Formatted string
    """
    lines = []
    
    # Answer
    answer = result.get("answer", "No answer generated.")
    lines.append(f"Answer:\n{answer}")
    
    # Sources
    sources = result.get("sources", [])
    lines.append(format_sources(sources))
    
    # Context (optional)
    if include_context:
        context = result.get("context", "")
        if context:
            lines.append(f"\nContext:\n{context[:500]}...")  # Truncate long context
    
    return "\n".join(lines)


def format_metadata(metadata: Dict[str, Any]) -> str:
    """
    Format metadata dictionary for display
    
    Args:
        metadata: Metadata dictionary
        
    Returns:
        Formatted string
    """
    if not metadata:
        return "No metadata"
    
    lines = []
    for key, value in metadata.items():
        if isinstance(value, (list, dict)):
            value = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
        lines.append(f"  {key}: {value}")
    
    return "\n".join(lines)


