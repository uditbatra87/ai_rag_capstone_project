from pydantic import Field
from pydantic_settings import BaseSettings,SettingsConfigDict

class Settings(BaseSettings):
    OPEN_AI_KEY : str
    OPEN_AI_URL : str
    OPEN_AI_RETRIES : int
    OPEN_AI_TIMEOUT : int

    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        extra="allow"
        )


my_settings = Settings()