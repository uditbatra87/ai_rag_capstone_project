from pydantic import Field
from pydantic_settings import BaseSettings,SettingsConfigDict

class Settings(BaseSettings):
    OPEN_AI_KEY : str
    OPEN_AI_URL : str

    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        extra="allow"
        )


my_settings = Settings()