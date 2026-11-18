"""Application configuration loaded from environment variables.

Centralises all settings (paths, model names, thresholds, API keys)
so the rest of the code never reads os.environ directly.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the RAG assistant.

    Values are read from environment variables or a local .env file.
    See .env.example for the full list of supported variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM provider ---
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    ollama_base_url: str = "http://localhost:11434"

    # --- Embeddings ---
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- ChromaDB ---
    chroma_db_path: str = "./chroma_db"
    chroma_collection_name: str = "student_handbook"

    # --- Retrieval ---
    retrieval_top_k: int = 20
    similarity_threshold: float = 0.3

    # --- Handbook PDF ---
    handbook_pdf_path: str = "data/student_handbook.pdf"

    @property
    def project_root(self) -> Path:
        """Absolute path to the project root (parent of the app package)."""
        return Path(__file__).resolve().parent.parent

    @property
    def handbook_pdf_absolute(self) -> Path:
        """Absolute path to the configured handbook PDF."""
        path = Path(self.handbook_pdf_path)
        return path if path.is_absolute() else self.project_root / path

    @property
    def chroma_db_absolute(self) -> Path:
        """Absolute path to the persistent ChromaDB directory."""
        path = Path(self.chroma_db_path)
        return path if path.is_absolute() else self.project_root / path


settings = Settings()
