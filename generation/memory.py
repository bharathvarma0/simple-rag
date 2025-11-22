"""
Conversational memory management
"""

from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from config import get_settings

@dataclass
class Message:
    role: str
    content: str

class ConversationMemory:
    """Manage conversational history"""
    
    def __init__(self):
        self.settings = get_settings()
        self.history: List[Message] = []
        self.window_size = self.settings.memory.history_window_size
        
    def add_turn(self, user_input: str, ai_output: str):
        """
        Add a conversation turn (user input + AI output)
        
        Args:
            user_input: User's message
            ai_output: AI's response
        """
        self.history.append(Message(role="user", content=user_input))
        self.history.append(Message(role="assistant", content=ai_output))
        
        # Trim history if needed
        if len(self.history) > self.window_size * 2:
            self.history = self.history[-(self.window_size * 2):]
            
    def get_history(self) -> List[Dict[str, str]]:
        """
        Get conversation history as list of dicts
        
        Returns:
            List of messages [{"role": "...", "content": "..."}]
        """
        return [asdict(msg) for msg in self.history]
    
    def get_history_string(self) -> str:
        """
        Get history formatted as a string for prompts
        
        Returns:
            String representation of history
        """
        formatted = []
        for msg in self.history:
            role = "User" if msg.role == "user" else "Assistant"
            formatted.append(f"{role}: {msg.content}")
        return "\n".join(formatted)
        
    def clear(self):
        """Clear conversation history"""
        self.history = []
