"""Runtime configuration, loaded from the environment (see .env.example)."""

from __future__ import annotations

from functools import lru_cache

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- database ---
    postgres_user: str = "sentinel"
    postgres_password: str = "sentinel"
    postgres_db: str = "bobsentinel"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url: str | None = None

    # --- credentials (all optional: the app starts without them and the
    # corresponding subsystem reports itself unconfigured on /healthz) ---
    cdse_username: str | None = None
    cdse_password: str | None = None
    cdse_s3_access_key: str | None = None
    cdse_s3_secret_key: str | None = None
    aisstream_api_key: str | None = None
    gfw_api_token: str | None = None

    # --- area of interest ---
    aoi_name: str = "Bangladesh EEZ"
    aoi_mrgid: int = 8481
    aoi_lat_min: float = 20.5
    aoi_lat_max: float = 22.8
    aoi_lon_min: float = 88.0
    aoi_lon_max: float = 92.7

    # --- fusion / detection ---
    match_radius_m: float = 500.0
    match_max_gap_s: int = 1800
    cfar_pfa: float = 1e-9
    cfar_guard: int = 8
    cfar_train: int = 24

    # --- app ---
    log_level: str = "INFO"
    data_dir: str = "data"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sqlalchemy_url(self) -> str:
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def aoi_bbox(self) -> tuple[float, float, float, float]:
        """(lon_min, lat_min, lon_max, lat_max) — the shapely/GeoJSON order."""
        return (self.aoi_lon_min, self.aoi_lat_min, self.aoi_lon_max, self.aoi_lat_max)

    def aisstream_bbox(self) -> list[list[list[float]]]:
        """aisstream.io wants [[[lat, lon], [lat, lon]]] — note lat first."""
        return [
            [
                [self.aoi_lat_min, self.aoi_lon_min],
                [self.aoi_lat_max, self.aoi_lon_max],
            ]
        ]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
