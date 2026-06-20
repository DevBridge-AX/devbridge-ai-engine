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

    # Embeddings
    # Day 2에서는 아직 강하게 사용하지 않음.
    # Day 3 이후 RAG 검색/문서 검색에서 사용 예정.
    embedding_model: str = "text-embedding-3-large"
    embedding_model_version: str = "v1"
    embedding_api_key: str = ""

    # Existing LLM provider settings
    # 기존 프로젝트 구조 보존용
    anthropic_api_key: str = ""
    main_model: str = "claude-sonnet-4-6"
    rewrite_model: str = "claude-haiku-4-5-20251001"

    # Day 2 document analysis mode
    # fallback: 실제 모델 API 호출 없이 규칙 기반 분석
    # openai: 실제 모델 API 호출
    ai_analysis_mode: str = "fallback"

    # Day 2 document analysis model
    # fallback 모드에서는 표시용 모델명으로만 사용
    # openai 모드에서는 실제 호출 모델명으로 사용
    document_analysis_model: str = "fallback-v1"

    # 실제 모델 API 사용 시 필요한 값
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    # provider.py 컨텍스트 길이 가드
    max_context_tokens: int = 30000

    # Spring backend 연동
    spring_backend_base_url: str = "http://localhost:8080"
    spring_user_lookup_path: str = "/internal/users/lookup"


@lru_cache
def get_settings() -> Settings:
    """Settings 싱글톤 인스턴스를 반환합니다."""
    return Settings()