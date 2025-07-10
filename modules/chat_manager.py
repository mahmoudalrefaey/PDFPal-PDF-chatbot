"""
Chat Manager Module
Handles conversation history and chat state management
"""

import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class ChatMessage:
    """Represents a single chat message"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

class ChatManager:
    """Manages chat history and conversation state"""
    
    def __init__(self, max_history: int = 100):
        """
        Initialize chat manager
        
        Args:
            max_history: Maximum number of messages to keep in history
        """
        self.max_history = max_history
        self.messages: List[ChatMessage] = []
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Add a new message to the chat history
        
        Args:
            role: Message role ('user' or 'assistant')
            content: Message content
            metadata: Optional metadata
        """
        try:
            # Create message
            message = ChatMessage(
                role=role,
                content=content,
                timestamp=datetime.now().isoformat(),
                metadata=metadata or {}
            )
            
            # Add to history
            self.messages.append(message)
            
            # Trim history if needed
            if len(self.messages) > self.max_history:
                self.messages = self.messages[-self.max_history:]
            
            self.logger.info(f"Added {role} message to chat history")
            
        except Exception as e:
            self.logger.error(f"Error adding message: {e}")
            raise
    
    def get_messages(self, limit: Optional[int] = None) -> List[ChatMessage]:
        """
        Get chat messages
        
        Args:
            limit: Maximum number of messages to return
            
        Returns:
            List of chat messages
        """
        if limit is None:
            return self.messages.copy()
        
        return self.messages[-limit:]
    
    def get_conversation_summary(self) -> str:
        """
        Get a summary of the conversation
        
        Returns:
            Conversation summary
        """
        if not self.messages:
            return "No conversation history"
        
        user_messages = [msg for msg in self.messages if msg.role == "user"]
        assistant_messages = [msg for msg in self.messages if msg.role == "assistant"]
        
        summary = f"Conversation Summary:\n"
        summary += f"- Total messages: {len(self.messages)}\n"
        summary += f"- User messages: {len(user_messages)}\n"
        summary += f"- Assistant messages: {len(assistant_messages)}\n"
        summary += f"- Started: {self.messages[0].timestamp}\n"
        summary += f"- Last activity: {self.messages[-1].timestamp}\n"
        
        return summary
    
    def clear_history(self):
        """Clear all chat history"""
        self.messages.clear()
        self.logger.info("Chat history cleared")
    
    def remove_message(self, index: int):
        """
        Remove a specific message by index
        
        Args:
            index: Index of message to remove
        """
        try:
            if 0 <= index < len(self.messages):
                removed = self.messages.pop(index)
                self.logger.info(f"Removed message at index {index}")
                return removed
            else:
                raise IndexError("Message index out of range")
        except Exception as e:
            self.logger.error(f"Error removing message: {e}")
            raise
    
    def get_context_for_rag(self, max_messages: int = 5) -> str:
        """
        Get recent conversation context for RAG
        
        Args:
            max_messages: Maximum number of recent messages to include
            
        Returns:
            Formatted conversation context
        """
        if not self.messages:
            return ""
        
        # Get recent messages
        recent_messages = self.messages[-max_messages:]
        
        # Format context
        context_parts = []
        for msg in recent_messages:
            role_label = "User" if msg.role == "user" else "Assistant"
            context_parts.append(f"{role_label}: {msg.content}")
        
        return "\n".join(context_parts)
    
    def save_conversation(self, filepath: str):
        """
        Save conversation to file
        
        Args:
            filepath: Path to save the conversation
        """
        try:
            # Convert messages to dictionaries
            messages_dict = [asdict(msg) for msg in self.messages]
            
            # Create conversation data
            conversation_data = {
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "total_messages": len(self.messages),
                    "max_history": self.max_history
                },
                "messages": messages_dict
            }
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Conversation saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving conversation: {e}")
            raise
    
    def load_conversation(self, filepath: str):
        """
        Load conversation from file
        
        Args:
            filepath: Path to load the conversation from
        """
        try:
            # Load from file
            with open(filepath, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)
            
            # Clear current history
            self.messages.clear()
            
            # Load messages
            for msg_data in conversation_data.get("messages", []):
                message = ChatMessage(
                    role=msg_data["role"],
                    content=msg_data["content"],
                    timestamp=msg_data["timestamp"],
                    metadata=msg_data.get("metadata")
                )
                self.messages.append(message)
            
            self.logger.info(f"Conversation loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading conversation: {e}")
            raise
    
    def export_conversation_text(self, filepath: str):
        """
        Export conversation as plain text
        
        Args:
            filepath: Path to save the text file
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("PDFPal Conversation Export\n")
                f.write("=" * 50 + "\n\n")
                
                for msg in self.messages:
                    role_label = "User" if msg.role == "user" else "Assistant"
                    timestamp = datetime.fromisoformat(msg.timestamp).strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{timestamp}] {role_label}:\n{msg.content}\n\n")
            
            self.logger.info(f"Conversation exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting conversation: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get chat statistics
        
        Returns:
            Dictionary with chat statistics
        """
        if not self.messages:
            return {
                "total_messages": 0,
                "user_messages": 0,
                "assistant_messages": 0,
                "avg_message_length": 0,
                "conversation_duration": "0:00:00"
            }
        
        user_messages = [msg for msg in self.messages if msg.role == "user"]
        assistant_messages = [msg for msg in self.messages if msg.role == "assistant"]
        
        # Calculate average message length
        total_length = sum(len(msg.content) for msg in self.messages)
        avg_length = total_length / len(self.messages) if self.messages else 0
        
        # Calculate conversation duration
        start_time = datetime.fromisoformat(self.messages[0].timestamp)
        end_time = datetime.fromisoformat(self.messages[-1].timestamp)
        duration = end_time - start_time
        
        return {
            "total_messages": len(self.messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "avg_message_length": round(avg_length, 2),
            "conversation_duration": str(duration),
            "start_time": self.messages[0].timestamp,
            "end_time": self.messages[-1].timestamp
        }
    
    def search_messages(self, query: str, case_sensitive: bool = False) -> List[ChatMessage]:
        """
        Search messages by content
        
        Args:
            query: Search query
            case_sensitive: Whether search should be case sensitive
            
        Returns:
            List of matching messages
        """
        matching_messages = []
        
        for msg in self.messages:
            content = msg.content
            if not case_sensitive:
                content = content.lower()
                query = query.lower()
            
            if query in content:
                matching_messages.append(msg)
        
        return matching_messages 