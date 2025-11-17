"""
Path utilities
"""

from pathlib import Path
from typing import Union, Optional


def get_project_root() -> Path:
    """
    Get project root directory
    
    Returns:
        Path to project root
    """
    # Assuming utils is in RAG-Tutorials/utils/
    # Project root is parent of utils
    return Path(__file__).parent.parent


def ensure_dir(path: Union[str, Path], create: bool = True) -> Path:
    """
    Ensure directory exists
    
    Args:
        path: Directory path
        create: Whether to create directory if it doesn't exist
        
    Returns:
        Path object
    """
    path_obj = Path(path)
    if create:
        path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_relative_path(path: Union[str, Path], base: Optional[Path] = None) -> Path:
    """
    Get relative path from base directory
    
    Args:
        path: Target path
        base: Base directory (default: project root)
        
    Returns:
        Relative path
    """
    if base is None:
        base = get_project_root()
    
    try:
        return Path(path).relative_to(base)
    except ValueError:
        return Path(path)

