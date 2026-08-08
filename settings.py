from pydantic import Field
from pydantic_settings import BaseSettings,SettingsConfigDict

class AppSettings(BaseSettings):
    open_ai_key : str
    open_ai_url : str


    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        extra="forbid"
        )


my_settings = AppSettings()