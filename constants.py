import os
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=os.environ.get('PERSIST_DIRECTORY'),
        anonymized_telemetry=False
)