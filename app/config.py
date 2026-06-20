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

    # LLM provider
    main_model: str = "claude-sonnet-4-6"
    rewrite_model: str = "gpt-5.4-nano"
    grounding_model: str = "gemini-3.5-flash-lite"

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

    # Spring backend 연동
    spring_backend_base_url: str = "http://localhost:8080"
    spring_user_lookup_path: str = "/internal/users/lookup"


@lru_cache
def get_settings() -> Settings:
    """Settings 싱글톤 인스턴스를 반환합니다."""
    return Settings()