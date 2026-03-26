"""Model-agnostic adapter layer for HallucinationGuard-Env.

This module provides unified interfaces for working with any LLM:
- OpenAI (GPT-3.5, GPT-4, etc.)
- Anthropic (Claude)
- HuggingFace models
- Ollama (local models)
- Generic API-compatible models

This makes the environment truly model-agnostic and impressive for production use.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
import os

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    """Unified response from any model."""
    answer: str
    confidence: float
    reasoning: str = ""
    source_quote: str = ""
    raw_response: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ModelConfig:
    """Configuration for model adapters."""
    model_name: str = ""
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 30
    extra_params: Dict[str, Any] = field(default_factory=dict)


class BaseModelAdapter(ABC):
    """Abstract base class for all model adapters."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._client = None

    @abstractmethod
    def generate_response(
        self,
        question: str,
        context: str,
        require_citation: bool = True,
        require_confidence: bool = True,
        require_reasoning: bool = False,
    ) -> ModelResponse:
        """Generate a response from the model."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is available and configured."""
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "model_name": self.config.model_name,
            "adapter_type": self.__class__.__name__,
            "available": self.is_available(),
        }


class OpenAIAdapter(BaseModelAdapter):
    """Adapter for OpenAI models (GPT-3.5, GPT-4, etc.)."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client = None

    def _get_client(self):
        """Lazy-load the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.config.api_key or os.getenv("OPENAI_API_KEY"),
                    base_url=self.config.api_base,
                    timeout=self.config.timeout,
                )
            except ImportError:
                logger.error("OpenAI package not installed. Install with: pip install openai")
                return None
        return self._client

    def is_available(self) -> bool:
        """Check if OpenAI is configured."""
        return bool(self.config.api_key or os.getenv("OPENAI_API_KEY"))

    def generate_response(
        self,
        question: str,
        context: str,
        require_citation: bool = True,
        require_confidence: bool = True,
        require_reasoning: bool = False,
    ) -> ModelResponse:
        """Generate response using OpenAI."""
        client = self._get_client()
        if not client:
            return ModelResponse(
                answer="",
                confidence=0.0,
                error="OpenAI client not available"
            )

        # Build prompt
        system_prompt = self._build_system_prompt(require_citation, require_confidence, require_reasoning)
        user_prompt = self._build_user_prompt(question, context)

        try:
            response = client.chat.completions.create(
                model=self.config.model_name or "gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **self.config.extra_params,
            )

            content = response.choices[0].message.content
            return self._parse_response(content, response)

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return ModelResponse(
                answer="",
                confidence=0.0,
                error=str(e)
            )

    def _build_system_prompt(self, require_citation: bool, require_confidence: bool, require_reasoning: bool) -> str:
        """Build the system prompt for OpenAI."""
        prompt = """You are an AI assistant that answers questions based ONLY on the provided context.

IMPORTANT RULES:
1. Answer using ONLY the information in the provided context
2. Do not use your prior knowledge or make up information
3. If the context doesn't contain the answer, say "I cannot find this information in the provided context"
4. Provide a direct quote from the context that supports your answer
5. Rate your confidence from 0.0 to 1.0"""

        if require_reasoning:
            prompt += "\n6. Explain your reasoning step by step"

        prompt += """

FORMAT YOUR RESPONSE AS JSON:
{
    "answer": "your answer here",
    "confidence": 0.95,
    "source_quote": "direct quote from context",
    "reasoning": "your reasoning here"
}"""

        return prompt

    def _build_user_prompt(self, question: str, context: str) -> str:
        """Build the user prompt."""
        return f"""CONTEXT:
{context}

QUESTION:
{question}

ANSWER (in JSON format):"""

    def _parse_response(self, content: str, raw_response: Any) -> ModelResponse:
        """Parse the model response."""
        try:
            # Try to parse as JSON
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            data = json.loads(content)

            return ModelResponse(
                answer=data.get("answer", ""),
                confidence=float(data.get("confidence", 0.5)),
                source_quote=data.get("source_quote", ""),
                reasoning=data.get("reasoning", ""),
                raw_response=raw_response,
                metadata={"model": raw_response.model if hasattr(raw_response, 'model') else ""}
            )
        except json.JSONDecodeError:
            # Fallback: extract from plain text
            return ModelResponse(
                answer=content,
                confidence=0.5,
                source_quote="",
                reasoning="",
                raw_response=raw_response,
                metadata={"parse_error": "Could not parse JSON"}
            )


class AnthropicAdapter(BaseModelAdapter):
    """Adapter for Anthropic models (Claude)."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client = None

    def _get_client(self):
        """Lazy-load the Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(
                    api_key=self.config.api_key or os.getenv("ANTHROPIC_API_KEY"),
                    timeout=self.config.timeout,
                )
            except ImportError:
                logger.error("Anthropic package not installed. Install with: pip install anthropic")
                return None
        return self._client

    def is_available(self) -> bool:
        """Check if Anthropic is configured."""
        return bool(self.config.api_key or os.getenv("ANTHROPIC_API_KEY"))

    def generate_response(
        self,
        question: str,
        context: str,
        require_citation: bool = True,
        require_confidence: bool = True,
        require_reasoning: bool = False,
    ) -> ModelResponse:
        """Generate response using Anthropic Claude."""
        client = self._get_client()
        if not client:
            return ModelResponse(
                answer="",
                confidence=0.0,
                error="Anthropic client not available"
            )

        system_prompt = """You are an AI assistant that answers questions based ONLY on the provided context.

IMPORTANT:
- Use ONLY information from the provided context
- Do not fabricate or hallucinate information
- Provide direct quotes to support your answer
- Rate your confidence from 0.0 to 1.0

Respond in JSON format with: answer, confidence, source_quote, and optionally reasoning."""

        user_prompt = f"""CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

        try:
            response = client.messages.create(
                model=self.config.model_name or "claude-sonnet-4-5-20250929",
                max_tokens=self.config.max_tokens,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
                **self.config.extra_params,
            )

            content = response.content[0].text if response.content else ""
            return self._parse_response(content, response)

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return ModelResponse(
                answer="",
                confidence=0.0,
                error=str(e)
            )

    def _parse_response(self, content: str, raw_response: Any) -> ModelResponse:
        """Parse the model response."""
        try:
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            data = json.loads(content)

            return ModelResponse(
                answer=data.get("answer", ""),
                confidence=float(data.get("confidence", 0.5)),
                source_quote=data.get("source_quote", ""),
                reasoning=data.get("reasoning", ""),
                raw_response=raw_response,
                metadata={"model": raw_response.model if hasattr(raw_response, 'model') else ""}
            )
        except json.JSONDecodeError:
            return ModelResponse(
                answer=content,
                confidence=0.5,
                source_quote="",
                reasoning="",
                raw_response=raw_response,
                metadata={"parse_error": "Could not parse JSON"}
            )


class HuggingFaceAdapter(BaseModelAdapter):
    """Adapter for HuggingFace models."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy-load the HuggingFace model."""
        if self._model is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch

                model_name = self.config.model_name or "mistralai/Mistral-7B-Instruct-v0.2"

                logger.info(f"Loading HuggingFace model: {model_name}")
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
                logger.info("Model loaded successfully")

            except ImportError:
                logger.error("Transformers package not installed. Install with: pip install transformers torch")
                return None
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                return None

        return self._model is not None

    def is_available(self) -> bool:
        """Check if HuggingFace model can be loaded."""
        try:
            from transformers import AutoModelForCausalLM
            return True
        except ImportError:
            return False

    def generate_response(
        self,
        question: str,
        context: str,
        require_citation: bool = True,
        require_confidence: bool = True,
        require_reasoning: bool = False,
    ) -> ModelResponse:
        """Generate response using HuggingFace model."""
        if not self._load_model():
            return ModelResponse(
                answer="",
                confidence=0.0,
                error="HuggingFace model not available"
            )

        try:
            import torch

            # Build prompt
            prompt = f"""Context: {context}

Question: {question}

Answer based ONLY on the context above. Provide:
1. Your answer
2. Confidence (0.0-1.0)
3. Source quote from context

JSON format: {{"answer": "...", "confidence": 0.9, "source_quote": "..."}}"""

            # Tokenize and generate
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    do_sample=self.config.temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            # Decode response
            generated_text = self._tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return self._parse_response(generated_text, outputs)

        except Exception as e:
            logger.error(f"HuggingFace generation error: {e}")
            return ModelResponse(
                answer="",
                confidence=0.0,
                error=str(e)
            )

    def _parse_response(self, content: str, raw_response: Any) -> ModelResponse:
        """Parse the model response."""
        try:
            # Try to extract JSON
            start_idx = content.find('{')
            if start_idx >= 0:
                end_idx = content.rfind('}') + 1
                if end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    data = json.loads(json_str)
                    return ModelResponse(
                        answer=data.get("answer", ""),
                        confidence=float(data.get("confidence", 0.5)),
                        source_quote=data.get("source_quote", ""),
                        reasoning=data.get("reasoning", ""),
                        raw_response=raw_response,
                    )
        except Exception:
            pass

        return ModelResponse(
            answer=content.strip(),
            confidence=0.5,
            source_quote="",
            reasoning="",
            raw_response=raw_response,
        )


class OllamaAdapter(BaseModelAdapter):
    """Adapter for Ollama (local models)."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._base_url = config.api_base or "http://localhost:11434"

    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            import requests
            response = requests.get(f"{self._base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def generate_response(
        self,
        question: str,
        context: str,
        require_citation: bool = True,
        require_confidence: bool = True,
        require_reasoning: bool = False,
    ) -> ModelResponse:
        """Generate response using Ollama."""
        try:
            import requests

            prompt = f"""Context: {context}

Question: {question}

Answer based ONLY on the context. Respond in JSON:
{{"answer": "...", "confidence": 0.9, "source_quote": "..."}}"""

            response = requests.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self.config.model_name or "llama2",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens,
                    }
                },
                timeout=self.config.timeout,
            )
            response.raise_for_status()

            result = response.json()
            content = result.get("response", "")
            return self._parse_response(content, result)

        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return ModelResponse(
                answer="",
                confidence=0.0,
                error=str(e)
            )

    def _parse_response(self, content: str, raw_response: Any) -> ModelResponse:
        """Parse the model response."""
        try:
            content = content.strip()
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                data = json.loads(json_str)
                return ModelResponse(
                    answer=data.get("answer", ""),
                    confidence=float(data.get("confidence", 0.5)),
                    source_quote=data.get("source_quote", ""),
                    reasoning=data.get("reasoning", ""),
                    raw_response=raw_response,
                )
        except Exception:
            pass

        return ModelResponse(
            answer=content,
            confidence=0.5,
            source_quote="",
            reasoning="",
            raw_response=raw_response,
        )


class GenericAPIAdapter(BaseModelAdapter):
    """Generic adapter for any OpenAI-compatible API."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client = None

    def _get_client(self):
        """Lazy-load the OpenAI client with custom base URL."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.config.api_key or "dummy-key",
                    base_url=self.config.api_base,
                    timeout=self.config.timeout,
                )
            except ImportError:
                logger.error("OpenAI package not installed")
                return None
        return self._client

    def is_available(self) -> bool:
        """Check if the API is available."""
        return bool(self.config.api_base)

    def generate_response(
        self,
        question: str,
        context: str,
        require_citation: bool = True,
        require_confidence: bool = True,
        require_reasoning: bool = False,
    ) -> ModelResponse:
        """Generate response using generic API."""
        client = self._get_client()
        if not client:
            return ModelResponse(
                answer="",
                confidence=0.0,
                error="Generic API client not available"
            )

        system_prompt = """Answer based ONLY on provided context. Respond in JSON:
{"answer": "...", "confidence": 0.9, "source_quote": "..."}}"""

        try:
            response = client.chat.completions.create(
                model=self.config.model_name or "default",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            content = response.choices[0].message.content
            return self._parse_response(content, response)

        except Exception as e:
            logger.error(f"Generic API error: {e}")
            return ModelResponse(
                answer="",
                confidence=0.0,
                error=str(e)
            )

    def _parse_response(self, content: str, raw_response: Any) -> ModelResponse:
        """Parse the model response."""
        try:
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]

            data = json.loads(content)
            return ModelResponse(
                answer=data.get("answer", ""),
                confidence=float(data.get("confidence", 0.5)),
                source_quote=data.get("source_quote", ""),
                reasoning=data.get("reasoning", ""),
                raw_response=raw_response,
            )
        except Exception:
            return ModelResponse(
                answer=content,
                confidence=0.5,
                source_quote="",
                reasoning="",
                raw_response=raw_response,
            )


class ModelAdapterFactory:
    """Factory for creating model adapters."""

    _adapters = {
        "openai": OpenAIAdapter,
        "anthropic": AnthropicAdapter,
        "huggingface": HuggingFaceAdapter,
        "ollama": OllamaAdapter,
        "generic": GenericAPIAdapter,
    }

    @classmethod
    def create(cls, model_type: str, config: ModelConfig) -> BaseModelAdapter:
        """Create a model adapter based on type."""
        adapter_class = cls._adapters.get(model_type.lower())
        if not adapter_class:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(cls._adapters.keys())}")
        return adapter_class(config)

    @classmethod
    def register_adapter(cls, name: str, adapter_class: type):
        """Register a custom adapter class."""
        cls._adapters[name.lower()] = adapter_class

    @classmethod
    def get_available_adapters(cls) -> List[str]:
        """Get list of available adapter types."""
        return list(cls._adapters.keys())


def create_adapter(
    model_type: str = "openai",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    **kwargs
) -> BaseModelAdapter:
    """
    Convenience function to create a model adapter.

    Args:
        model_type: Type of adapter (openai, anthropic, huggingface, ollama, generic)
        model_name: Specific model name
        api_key: API key for the service
        api_base: Base URL for API (for generic/ollama)
        **kwargs: Additional configuration

    Returns:
        Configured model adapter

    Examples:
        # OpenAI
        adapter = create_adapter("openai", model_name="gpt-4", api_key="sk-...")

        # Anthropic
        adapter = create_adapter("anthropic", model_name="claude-3-opus", api_key="sk-ant-...")

        # Ollama (local)
        adapter = create_adapter("ollama", model_name="llama2")

        # HuggingFace
        adapter = create_adapter("huggingface", model_name="mistralai/Mistral-7B-Instruct-v0.2")
    """
    config = ModelConfig(
        model_name=model_name,
        api_key=api_key,
        api_base=api_base,
        **kwargs
    )
    return ModelAdapterFactory.create(model_type, config)
