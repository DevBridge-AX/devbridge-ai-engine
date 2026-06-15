"""
LLM API 클라이언트 추상화.

외부 LLM provider(구체 모델 미정)를 pluggable하게 사용하기 위한 인터페이스입니다.
query_rewriter, persona 변환, grounding의 structured output 호출 등 모든 LLM 호출은
이 인터페이스를 통해서만 이루어져야 합니다.

TODO:
- LLMProvider 추상 베이스 클래스 메서드 확정 (generate_stream, generate_structured 등)
- placeholder provider 구현체 추가 (config.llm_model_name 등 사용)
- 실제 모델 바인딩은 추후 결정
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any


class LLMProvider(ABC):
    """LLM provider 추상 인터페이스. 구체 모델은 placeholder."""

    @abstractmethod
    async def generate_stream(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> AsyncIterator[str]:
        """토큰 단위 스트리밍 응답 생성. TODO: 구현."""
        raise NotImplementedError

    @abstractmethod
    async def generate_structured(
        self, messages: list[dict[str, str]], schema: type, **kwargs: Any
    ) -> Any:
        """structured output(JSON) 생성 - grounding 등에서 사용. TODO: 구현."""
        raise NotImplementedError


# TODO: PlaceholderLLMProvider(LLMProvider) 구현체 추가
