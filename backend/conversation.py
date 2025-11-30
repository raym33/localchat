"""Conversation management for English practice."""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a friendly and patient English conversation partner helping someone practice their English speaking skills. Your role is to:

1. Engage in natural, flowing conversations on various topics
2. Respond in clear, natural English that's appropriate for conversation practice
3. Keep responses concise (1-3 sentences typically) to maintain a natural dialogue flow
4. Gently correct significant grammar or vocabulary mistakes by rephrasing correctly
5. Ask follow-up questions to keep the conversation going
6. Adapt your vocabulary and complexity to the user's apparent level
7. Be encouraging and supportive

Topics you can discuss include: daily life, hobbies, travel, food, work, culture, current events, hypothetical scenarios, opinions, and more.

Remember: This is spoken conversation practice, so keep responses natural and conversational, not formal or written-style. Do NOT use markdown formatting, asterisks, or special characters - just plain conversational text."""


@dataclass
class Message:
    """A conversation message."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Conversation:
    """A conversation session."""

    id: str
    messages: list[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


class ConversationManager:
    """Manages conversations and generates AI responses using LM Studio."""

    def __init__(self, lm_studio_url: str = "http://10.183.140.67:1234", model: str = "qwen3-4b-thinking-2507"):
        self.lm_studio_url = lm_studio_url.rstrip("/")
        self.model = model
        self.conversations: dict[str, Conversation] = {}
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    def create_conversation(self, conversation_id: str) -> Conversation:
        """Create a new conversation."""
        conv = Conversation(id=conversation_id)
        self.conversations[conversation_id] = conv
        return conv

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get an existing conversation."""
        return self.conversations.get(conversation_id)

    def add_message(
        self, conversation_id: str, role: str, content: str
    ) -> Optional[Message]:
        """Add a message to a conversation."""
        conv = self.get_conversation(conversation_id)
        if conv is None:
            conv = self.create_conversation(conversation_id)

        message = Message(role=role, content=content)
        conv.messages.append(message)
        return message

    def _clean_response(self, text: str) -> str:
        """Clean the response from thinking tags and markdown.

        Qwen thinking models output: <think>...reasoning...</think>actual response
        We only want the actual response after </think>
        """
        # Extract content AFTER </think> - this is the actual response
        if '</think>' in text:
            text = text.split('</think>')[-1]
        # If there's only <think> without </think>, the model is still thinking
        # In this case, there's no real response yet - return empty or wait
        elif '<think>' in text:
            # No response yet, just thinking
            text = ""

        # Same for <thinking> tags
        if '</thinking>' in text:
            text = text.split('</thinking>')[-1]
        elif '<thinking>' in text:
            text = ""

        # Remove any remaining tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove markdown formatting
        text = re.sub(r'\*+', '', text)
        text = re.sub(r'(?<![a-zA-Z])_+|_+(?![a-zA-Z])', ' ', text)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'`[^`]+`', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.strip()

    async def generate_response(
        self, conversation_id: str, user_message: str
    ) -> str:
        """Generate an AI response to the user's message.

        Args:
            conversation_id: The conversation ID
            user_message: The user's message

        Returns:
            The AI's response text
        """
        # Add user message to history
        self.add_message(conversation_id, "user", user_message)

        conv = self.get_conversation(conversation_id)

        try:
            # Build messages for API
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

            # Add conversation history (last 20 messages for context)
            for msg in conv.messages[-20:]:
                messages.append({"role": msg.role, "content": msg.content})

            # Call LM Studio API (OpenAI-compatible endpoint)
            # Note: thinking models need more tokens for <think>...</think> + response
            response = await self.client.post(
                f"{self.lm_studio_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 1024,
                    "temperature": 0.8,
                    "stream": False,
                },
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

            data = response.json()
            raw_response = data["choices"][0]["message"]["content"]

            # Clean the response
            cleaned_response = self._clean_response(raw_response)

            self.add_message(conversation_id, "assistant", cleaned_response)
            return cleaned_response

        except httpx.ConnectError as e:
            logger.error(f"Cannot connect to LM Studio at {self.lm_studio_url}: {e}")
            response = self._get_fallback_response(user_message, conv)
            self.add_message(conversation_id, "assistant", response)
            return response
        except Exception as e:
            logger.error(f"Error calling LM Studio API: {e}")
            response = self._get_fallback_response(user_message, conv)
            self.add_message(conversation_id, "assistant", response)
            return response

    def _get_fallback_response(
        self, user_message: str, conv: Conversation
    ) -> str:
        """Get a fallback response when LM Studio is not available."""
        user_lower = user_message.lower()

        # Greeting responses
        if any(
            greeting in user_lower
            for greeting in ["hello", "hi", "hey", "good morning", "good afternoon"]
        ):
            responses = [
                "Hello! It's great to practice English with you today. What would you like to talk about?",
                "Hi there! How are you doing today? I'd love to hear about your day.",
                "Hey! Nice to meet you. What topics interest you for our conversation?",
            ]
            idx = len(conv.messages) % len(responses)
            return responses[idx]

        # Questions about the user
        if "how are you" in user_lower:
            return "I'm doing well, thank you for asking! How about you? How has your day been so far?"

        # Hobby-related
        if any(word in user_lower for word in ["hobby", "hobbies", "like to do", "free time"]):
            return "That's a great topic! Hobbies are so important for relaxation. What do you enjoy doing in your free time?"

        # Weather
        if "weather" in user_lower:
            return "Weather is always an interesting topic! How's the weather where you are? Do you prefer sunny or rainy days?"

        # Food
        if any(word in user_lower for word in ["food", "eat", "restaurant", "cook"]):
            return "I love talking about food! What's your favorite cuisine? Do you enjoy cooking or do you prefer eating out?"

        # Travel
        if any(word in user_lower for word in ["travel", "trip", "vacation", "visit"]):
            return "Traveling is wonderful for learning about different cultures. Have you traveled anywhere interesting recently, or is there a place you dream of visiting?"

        # Work/study
        if any(word in user_lower for word in ["work", "job", "study", "school", "university"]):
            return "That's an interesting topic. What field do you work in or study? What do you enjoy most about it?"

        # Default conversational responses
        default_responses = [
            "That's interesting! Can you tell me more about that?",
            "I see! What made you think about this topic?",
            "That's a great point. How do you feel about it?",
            "Interesting perspective! Do you have any examples?",
            "I'd love to hear more. What else can you share about this?",
        ]
        idx = len(conv.messages) % len(default_responses)
        return default_responses[idx]

    def clear_conversation(self, conversation_id: str) -> None:
        """Clear a conversation's history."""
        if conversation_id in self.conversations:
            self.conversations[conversation_id].messages.clear()

    def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation."""
        self.conversations.pop(conversation_id, None)

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Global instance
_conversation_manager: Optional[ConversationManager] = None


def get_conversation_manager() -> ConversationManager:
    """Get or create the conversation manager."""
    global _conversation_manager
    if _conversation_manager is None:
        from .config import config

        _conversation_manager = ConversationManager(
            lm_studio_url=config.lm_studio_url,
            model=config.lm_studio_model,
        )
    return _conversation_manager
