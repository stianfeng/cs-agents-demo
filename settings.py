from dotenv import find_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        case_sensitive=True,
        env_parse_none_str="None",
        env_ignore_empty=True,
        env_nested_delimiter="__",
        env_nested_max_split=1,
    )
    # OpenAI API key
    OPENAI_API_KEY: SecretStr = ""
    OPENAI_MODEL: str = "gpt-4.1-mini"
    
    # FastAPI host and port
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8080

settings = Settings()