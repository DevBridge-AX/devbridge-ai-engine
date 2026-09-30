"""
앱 전역 설정.

환경 변수(.env)를 로드하여 DB, 벡터스토어, 임베딩, LLM provider, Spring 연동 등에
필요한 설정값을 제공합니다. pydantic-settings의 BaseSettings를 사용합니다.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "local"
    log_level: str = "INFO"

    # Internal API 인증
    # Spring Backend -> FastAPI AI Engine 호출 시 X-Internal-Api-Key 헤더로 검증
    internal_api_key: str = "changeme"

    # Database
    database_url: str = "mysql+pymysql://ai_engine_user:changeme@localhost:3306/devbridge_ai"

    # Vector store
    vector_store_provider: str = "chroma"
    vector_store_path: str = "./data/vector_store"

    # GMS 통합 API Key
    # Claude / GPT / Gemini 호출에 공통으로 사용
    gms_api_key: str = ""

    # GMS Provider Base URLs
    # provider.py가 모델명 prefix로 provider를 자동 감지하여 해당 URL을 사용합니다.
    anthropic_base_url: str = "https://gms.ssafy.io/gmsapi/api.anthropic.com"
    openai_base_url: str = "https://gms.ssafy.io/gmsapi/api.openai.com/v1"
    gemini_base_url: str = "https://gms.ssafy.io/gmsapi/generativelanguage.googleapis.com/v1beta"

    # Embeddings
    # Gemini Embedding via GMS
    embedding_model: str = "gemini-embedding-2"
    embedding_model_version: str = "v1"

    # Gemini Embedding API는 응답에 실측 토큰 수를 포함하지 않으므로, GMS 과금 단위
    # 기준으로 텍스트 1건당 0.2 토큰을 고정 추정치로 사용합니다(usage_logs 누적용).
    # 실측이 아닌 추정치이며, embedder.EmbedResult.token_source="estimate_per_text"로 표시됩니다.
    embedding_tokens_per_text: float = 0.2

    # LLM provider
    main_model: str = "claude-sonnet-4-6"
    # 기본값 근거: docs/rewrite-model-eval.md (A5 평가, 품질 동률·지연 약 2.1배 단축)
    rewrite_model: str = "claude-haiku-4-5-20251001"
    grounding_model: str = "gemini-2.5-flash-lite"

    # 메인 답변 생성 스트리밍 호출의 max_tokens (chat_pipeline.run() → call_main_stream()).
    # provider.call_main_stream()의 기본값(4096)과 동일하므로 기본 동작은 변하지 않습니다.
    # 라이브 검증 하네스(tests/live)는 비용 상한을 위해 이 값을 낮춰 override합니다.
    main_max_tokens: int = 4096

    # Day 2 document analysis mode
    # fallback: 실제 모델 API 호출 없이 규칙 기반 분석
    # llm: provider.py의 GMS LLM 호출 구조 사용
    ai_analysis_mode: str = "fallback"

    # Day 2 document analysis model
    # fallback 모드에서는 표시용 모델명으로 사용
    # llm 모드에서는 실제 호출 모델명으로 사용
    document_analysis_model: str = "fallback-v1"

    # provider.py 컨텍스트 길이 가드
    max_context_tokens: int = 30000

    # 그라운딩 유사도 1차 필터 임계치
    grounding_similarity_threshold: float = 0.35

    # 그라운딩 2차 판정 시스템 프롬프트 버전 (app/core/rag/grounding_prompts.py의 키)
    grounding_prompt_version: str = "v1"
    # 2차 판정 프롬프트에 넣는 청크 수 (retrieve top_k와 별개, 앞에서부터 사용)
    grounding_judge_top_k: int = 5
    # 0이면 제한 없음. 양수면 판정 프롬프트의 청크 content를 이 길이로 자름
    grounding_judge_max_chunk_chars: int = 0

    # BM25 하이브리드 검색
    bm25_enabled: bool = True
    bm25_cache_ttl_seconds: int = 600

    # Spring backend 연동
    spring_backend_base_url: str = "http://localhost:8080"
    spring_user_lookup_path: str = "/internal/users/lookup"

    # Observability metrics (app/core/metrics.py)
    # 요청/인덱싱 1건당 JSONL 1줄을 {metrics_dir}/{event}.jsonl에 append합니다.
    # 파일 쓰기 실패는 요청 흐름에 영향을 주지 않도록 경고 로그만 남기고 무시합니다.
    metrics_enabled: bool = True
    metrics_dir: str = "./data/metrics"

    # 인덱싱 파싱 판정: read_text(errors="replace") 치환 문자(�) 비율이 이 값을
    # 초과하면 메트릭상 result=PARSE_WARN으로 구분합니다(analysis_status는 불변).
    ingestion_parse_warn_ratio: float = 0.05

    # LoRA 학습데이터 export 출력 루트 ({dir}/{workspace_id}/{dataset_version}.jsonl)
    training_data_dir: str = "./data/training"


@lru_cache
def get_settings() -> Settings:
    """Settings 싱글톤 인스턴스를 반환합니다."""
    return Settings()