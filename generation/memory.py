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
    """Manage conversational history with summarization"""
    
    def __init__(self):
        self.settings = get_settings()
        self.history: List[Message] = []
        self.window_size = self.settings.memory.history_window_size
        self.summary = ""  # Long-term summary of conversation
        
    def add_turn(self, user_input: str, ai_output: str):
        """
        Add a conversation turn (user input + AI output)
        
        Args:
            user_input: User's message
            ai_output: AI's response
        """
        self.history.append(Message(role="user", content=user_input))
        self.history.append(Message(role="assistant", content=ai_output))
        
        # Trim history if needed (keep slightly more than window for summarization context)
        # We don't hard trim here anymore, we rely on summarize() to manage history size
        # But as a safeguard against infinite growth if summarize isn't called:
        max_history = 20
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]
            
    def get_history(self) -> List[Dict[str, str]]:
        """
        Get conversation history as list of dicts
        
        Args:
            Returns:
            List of messages [{"role": "...", "content": "..."}]
        """
        return [asdict(msg) for msg in self.history]
    
    def get_history_string(self) -> str:
        """
        Get history formatted as a string for prompts
        Includes summary + recent turns
        
        Returns:
            String representation of history
        """
        formatted = []
        
        if self.summary:
            formatted.append(f"Summary of previous conversation:\n{self.summary}\n")
            
        formatted.append("Recent conversation:")
        # Return last N turns based on window size
        recent_history = self.history[-(self.window_size * 2):] if self.history else []
        
        for msg in recent_history:
            role = "User" if msg.role == "user" else "Assistant"
            formatted.append(f"{role}: {msg.content}")
            
        return "\n".join(formatted)
        
    def summarize(self, llm_wrapper: Any):
        """
        Summarize older conversation turns
        
        Args:
            llm_wrapper: LLMWrapper instance to generate summary
        """
        # Only summarize if we have enough history (e.g., > window_size * 2)
        if len(self.history) <= self.window_size * 2:
            return
            
        # Get turns to summarize (everything except the last window_size turns)
        turns_to_summarize = self.history[:-(self.window_size * 2)]
        recent_turns = self.history[-(self.window_size * 2):]
        
        if not turns_to_summarize:
            return
            
        # Format turns for summarization
        text_to_summarize = ""
        for msg in turns_to_summarize:
            role = "User" if msg.role == "user" else "Assistant"
            text_to_summarize += f"{role}: {msg.content}\n"
            
        # Create prompt
        prompt = f"""
        Please summarize the following conversation concisely, focusing on key entities, user intent, and important details.
        If there is an existing summary, integrate the new information into it.
        
        Existing Summary:
        {self.summary}
        
        New Conversation to Add:
        {text_to_summarize}
        
        Updated Summary:
        """
        
        try:
            # Generate new summary
            new_summary = llm_wrapper.generate(prompt).strip()
            self.summary = new_summary
            
            # Update history to only keep recent turns
            self.history = recent_turns
            
        except Exception as e:
            print(f"[ERROR] Failed to summarize conversation: {e}")

    def clear(self):
        """Clear conversation history"""
        self.history = []
        self.summary = ""
