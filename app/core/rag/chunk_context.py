"""
청크 문맥 보강(Contextual Retrieval) 생성기.

문서 전문과 청크 1건을 경량 LLM에 보내 "이 청크가 문서 안에서 어떤 맥락인지" 1~2문장을 만듭니다.
생성된 문맥은 임베딩 입력 앞에만 붙으며, document_chunks.content(원문)는 바꾸지 않습니다.

- 청크별 sha1(content)가 reuse 맵에 있으면 LLM을 호출하지 않고 기존 문맥을 재사용합니다.
- 실패(타임아웃·오류·빈 응답)한 청크는 None(fail-open)으로 두고 failed로 집계합니다.
- 429(httpx.HTTPStatusError)에 한해 settings.chunk_context_max_retries회까지 15·30·60초 백오프 재시도합니다.
- 토큰 사용량은 call_with_model이 llm_calls(purpose=chunk_context)에 기록하며 usage_logs에는 넣지 않습니다.
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass

import httpx

from app.config import get_settings
from app.core.llm.provider import call_with_model
from app.core.rag.chunker import Chunk

logger = logging.getLogger(__name__)

PURPOSE = "chunk_context"
MAX_CONTEXT_CHARS = 300
CALL_TIMEOUT_SECONDS = 10.0
_BACKOFF_SECONDS = (15, 30, 60)
_TRUNCATION_MARK = "…(생략)"

SYSTEM_PROMPT = (
    "문서 전체와 그중 한 문단을 준다. 검색에 도움이 되도록 이 문단이 문서 안에서 어떤 맥락인지 "
    "1~2문장으로 설명하라. 문단 내용을 반복하지 말고, 문서 제목·절 이름·대상(환경, 팀, 시스템)을 "
    "포함하라. 설명만 출력."
)


@dataclass
class ChunkContextResult:
    contexts: list[str | None]  # chunks와 같은 길이, 실패/비활성은 None
    llm_calls: int
    failed: int
    reused: int
    retried: int
    latency_ms: float
    usage_prompt_tokens: int
    usage_completion_tokens: int


def content_sha1(content: str) -> str:
    """청크 원문의 sha1 hex. 재인덱싱 시 문맥 재사용 키로 쓴다."""
    return hashlib.sha1(content.encode("utf-8")).hexdigest()


def _truncate_doc(doc_text: str, max_chars: int) -> str:
    if len(doc_text) <= max_chars:
        return doc_text
    return doc_text[:max_chars] + _TRUNCATION_MARK


async def generate_chunk_contexts(
    doc_title: str,
    doc_text: str,
    chunks: list[Chunk],
    *,
    reuse: dict[str, str] | None = None,
) -> ChunkContextResult:
    """청크마다 문맥을 생성해 chunks와 같은 길이의 contexts를 반환합니다."""
    settings = get_settings()
    start = time.perf_counter()
    contexts: list[str | None] = [None] * len(chunks)

    if not settings.chunk_context_enabled or not chunks:
        return ChunkContextResult(contexts, 0, 0, 0, 0, 0.0, 0, 0)

    reuse = reuse or {}
    doc_part = _truncate_doc(doc_text, settings.chunk_context_max_doc_chars)
    semaphore = asyncio.Semaphore(settings.chunk_context_concurrency)
    max_retries = settings.chunk_context_max_retries

    stats = {"llm_calls": 0, "retried": 0, "prompt": 0, "completion": 0}

    async def _one(content: str) -> str | None:
        user = (
            f"문서 제목: {doc_title}\n\n<document>\n{doc_part}\n</document>\n\n"
            f"<chunk>\n{content}\n</chunk>"
        )
        attempt = 0
        async with semaphore:
            while True:
                stats["llm_calls"] += 1
                try:
                    text, usage = await call_with_model(
                        settings.chunk_context_model,
                        [{"role": "user", "content": user}],
                        SYSTEM_PROMPT,
                        max_tokens=120,
                        purpose=PURPOSE,
                        timeout=CALL_TIMEOUT_SECONDS,
                    )
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 429 and attempt < max_retries:
                        delay = _BACKOFF_SECONDS[min(attempt, len(_BACKOFF_SECONDS) - 1)]
                        attempt += 1
                        stats["retried"] += 1
                        await asyncio.sleep(delay)
                        continue
                    raise
                stats["prompt"] += usage.prompt_tokens
                stats["completion"] += usage.completion_tokens
                return (text or "").strip()[:MAX_CONTEXT_CHARS] or None

    reused = 0
    tasks: list[tuple[int, "asyncio.Task[str | None]"]] = []
    for i, chunk in enumerate(chunks):
        cached = reuse.get(content_sha1(chunk.content))
        if cached:
            contexts[i] = cached
            reused += 1
        else:
            tasks.append((i, asyncio.ensure_future(_one(chunk.content))))

    results = await asyncio.gather(*(t for _, t in tasks), return_exceptions=True)
    failed = 0
    for (i, _), res in zip(tasks, results):
        if isinstance(res, asyncio.CancelledError):
            raise res
        if isinstance(res, BaseException) or not res:
            failed += 1
            if isinstance(res, BaseException):
                logger.warning("chunk_context failed: %s", type(res).__name__)
            continue
        contexts[i] = res

    return ChunkContextResult(
        contexts=contexts,
        llm_calls=stats["llm_calls"],
        failed=failed,
        reused=reused,
        retried=stats["retried"],
        latency_ms=(time.perf_counter() - start) * 1000,
        usage_prompt_tokens=stats["prompt"],
        usage_completion_tokens=stats["completion"],
    )
