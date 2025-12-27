"""
Redis caching utility
"""

import redis
import json
import hashlib
from functools import wraps
from typing import Optional, Any
from config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

class RedisCache:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisCache, cls).__new__(cls)
            cls._instance.client = None
            cls._instance.enabled = False
            cls._instance.initialized = False
        return cls._instance
    
    def initialize(self):
        if self.initialized:
            return
            
        settings = get_settings()
        if settings.cache.enabled:
            try:
                self.client = redis.from_url(settings.cache.redis_url)
                self.client.ping()
                self.enabled = True
                self.ttl = settings.cache.ttl
                self.initialized = True
                logger.info(f"Redis cache initialized at {settings.cache.redis_url}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Caching disabled.")
                self.enabled = False
                self.initialized = True
    
    def get(self, key: str) -> Optional[Any]:
        if not self.initialized:
            self.initialize()
            
        if not self.enabled:
            return None
        try:
            data = self.client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        if not self.initialized:
            self.initialize()
            
        if not self.enabled:
            return
        try:
            self.client.setex(
                key, 
                ttl or self.ttl, 
                json.dumps(value, default=str)
            )
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")

def cache_result(prefix: str = "cache", ttl: Optional[int] = None):
    """Decorator to cache function results in Redis"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Initialize cache if needed
                cache = RedisCache()
                
                # Generate cache key
                # Skip 'self' (first arg) if it's a method
                clean_args = args[1:] if args and hasattr(args[0], '__class__') else args
                
                arg_str = json.dumps(clean_args, default=str)
                kwarg_str = json.dumps(kwargs, default=str)
                key_hash = hashlib.md5(f"{arg_str}{kwarg_str}".encode()).hexdigest()
                key = f"{prefix}:{func.__name__}:{key_hash}"
                
                # Check cache
                cached_val = cache.get(key)
                if cached_val:
                    logger.debug(f"Cache hit for {key}")
                    return cached_val
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Cache result
                cache.set(key, result, ttl)
                return result
                
            except Exception as e:
                # If caching logic fails, just run the function
                logger.warning(f"Caching wrapper error: {e}")
                return func(*args, **kwargs)
                
        return wrapper
    return decorator
