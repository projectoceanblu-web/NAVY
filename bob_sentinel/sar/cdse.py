"""Copernicus Data Space Ecosystem client — auth, catalogue search, download.

Auth is the OAuth2 *password* grant against the CDSE Keycloak realm; access
tokens live ~10 minutes, so every request goes through :meth:`CDSEClient.token`
which refreshes ahead of expiry (using the refresh token when it can, falling
back to a full re-login).

Search is OData rather than STAC: the ``$filter`` grammar expresses
"IW_GRDH_1S intersecting this polygon in this window" directly, and the STAC
endpoint moved to https://stac.dataspace.copernicus.eu/v1 in Nov 2025 while the
OData endpoint stayed put.

Constellation note (2026): Sentinel-1A concluded operations on 30 June 2026 and
Sentinel-1D is fully operational, so the default platform filter is S1C + S1D.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu"
    "/auth/realms/CDSE/protocol/openid-connect/token"
)
ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1"
STAC_URL = "https://stac.dataspace.copernicus.eu/v1"
ZIPPER_URL = "https://zipper.dataspace.copernicus.eu/odata/v1"
S3_ENDPOINT = "https://eodata.dataspace.copernicus.eu"

PUBLIC_CLIENT_ID = "cdse-public"

#: Sentinel-1A retired 2026-06-30; 1B retired 2021.  Build against 1C/1D.
ACTIVE_PLATFORMS = ("S1C", "S1D")

#: Refresh this many seconds before the token actually expires.
_TOKEN_SKEW_S = 60


class CDSEError(RuntimeError):
    """Any non-recoverable failure talking to CDSE."""


@dataclass
class Product:
    """One catalogue entry (a Sentinel-1 product)."""

    id: str
    name: str
    footprint_wkt: str | None
    acquired_start: datetime
    acquired_end: datetime
    size_bytes: int | None = None
    online: bool = True
    attributes: dict[str, object] = field(default_factory=dict)

    @property
    def platform(self) -> str | None:
        """S1A/S1C/S1D, parsed from the product name."""
        return self.name[:3].upper() if len(self.name) >= 3 else None

    @property
    def product_type(self) -> str | None:
        value = self.attributes.get("productType")
        return str(value) if value is not None else None

    @classmethod
    def from_odata(cls, item: dict) -> Product:
        attributes = {
            a.get("Name"): a.get("Value")
            for a in item.get("Attributes", []) or []
            if isinstance(a, dict)
        }
        footprint = item.get("Footprint")
        if isinstance(footprint, str) and footprint.startswith("geography'"):
            # "geography'SRID=4326;POLYGON((...))'" -> "POLYGON((...))"
            footprint = footprint.split(";", 1)[-1].rstrip("'")
        return cls(
            id=str(item["Id"]),
            name=str(item["Name"]),
            footprint_wkt=footprint,
            acquired_start=_parse_dt(item.get("ContentDate", {}).get("Start")),
            acquired_end=_parse_dt(item.get("ContentDate", {}).get("End")),
            size_bytes=_as_int(item.get("ContentLength")),
            online=bool(item.get("Online", True)),
            attributes=attributes,
        )


def _parse_dt(value: str | None) -> datetime:
    if not value:
        raise CDSEError("product is missing a ContentDate")
    # CDSE returns e.g. 2026-01-03T11:42:17.123456Z — Python 3.11 handles 'Z'.
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _as_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def build_odata_filter(
    *,
    collection: str = "SENTINEL-1",
    product_type: str = "IW_GRDH_1S",
    start: datetime,
    end: datetime,
    wkt: str | None = None,
    platforms: Sequence[str] | None = ACTIVE_PLATFORMS,
) -> str:
    """Compose the OData ``$filter`` string for a Sentinel-1 search.

    Kept separate from the HTTP call so it can be unit-tested without network.
    """
    clauses = [
        f"Collection/Name eq '{collection}'",
        (
            "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
            f"and att/OData.CSC.StringAttribute/Value eq '{product_type}')"
        ),
        f"ContentDate/Start ge {_odata_dt(start)}",
        f"ContentDate/Start le {_odata_dt(end)}",
    ]
    if wkt:
        clauses.append(f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')")
    if platforms:
        # Product names are prefixed S1A_/S1C_/S1D_ — cheaper than an attribute
        # filter and it is the only place the platform appears reliably.
        platform_clause = " or ".join(
            f"startswith(Name,'{p.upper()}_')" for p in platforms
        )
        clauses.append(f"({platform_clause})")
    return " and ".join(clauses)


def _odata_dt(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")


class CDSEClient:
    """Thin, dependency-light CDSE client (auth + OData search + download)."""

    def __init__(
        self,
        username: str,
        password: str,
        *,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not username or not password:
            raise CDSEError(
                "CDSE credentials missing — set CDSE_USERNAME / CDSE_PASSWORD"
            )
        self._username = username
        self._password = password
        self._client = client or httpx.Client(
            timeout=timeout, follow_redirects=True, headers={"User-Agent": "bob-sentinel"}
        )
        self._access_token: str | None = None
        self._refresh_token: str | None = None
        self._expires_at: float = 0.0

    # -- auth ---------------------------------------------------------------

    @property
    def token(self) -> str:
        """A valid access token, refreshed transparently."""
        if self._access_token and time.time() < self._expires_at - _TOKEN_SKEW_S:
            return self._access_token
        if self._refresh_token:
            try:
                return self._grant(
                    {
                        "client_id": PUBLIC_CLIENT_ID,
                        "grant_type": "refresh_token",
                        "refresh_token": self._refresh_token,
                    }
                )
            except CDSEError:
                log.info("CDSE refresh token rejected; re-authenticating")
        return self._grant(
            {
                "client_id": PUBLIC_CLIENT_ID,
                "grant_type": "password",
                "username": self._username,
                "password": self._password,
            }
        )

    def _grant(self, data: dict[str, str]) -> str:
        response = self._client.post(TOKEN_URL, data=data)
        if response.status_code != 200:
            raise CDSEError(
                f"CDSE token request failed ({response.status_code}): {response.text[:200]}"
            )
        payload = response.json()
        self._access_token = payload["access_token"]
        self._refresh_token = payload.get("refresh_token", self._refresh_token)
        self._expires_at = time.time() + float(payload.get("expires_in", 600))
        return self._access_token

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    # -- catalogue ----------------------------------------------------------

    def search(
        self,
        *,
        start: datetime,
        end: datetime,
        wkt: str | None = None,
        product_type: str = "IW_GRDH_1S",
        platforms: Sequence[str] | None = ACTIVE_PLATFORMS,
        limit: int = 100,
        page_size: int = 50,
    ) -> list[Product]:
        """Search the OData catalogue, following ``@odata.nextLink`` pages."""
        params = {
            "$filter": build_odata_filter(
                product_type=product_type,
                start=start,
                end=end,
                wkt=wkt,
                platforms=platforms,
            ),
            "$orderby": "ContentDate/Start desc",
            "$expand": "Attributes",
            "$top": str(min(page_size, limit)),
        }
        url: str | None = f"{ODATA_URL}/Products"
        products: list[Product] = []
        while url and len(products) < limit:
            response = self._client.get(url, params=params if params else None)
            if response.status_code != 200:
                raise CDSEError(
                    f"CDSE search failed ({response.status_code}): {response.text[:200]}"
                )
            payload = response.json()
            for item in payload.get("value", []):
                products.append(Product.from_odata(item))
                if len(products) >= limit:
                    break
            url = payload.get("@odata.nextLink")
            params = {}  # nextLink already carries the query string
        return products

    # -- download -----------------------------------------------------------

    def download(
        self,
        product: Product | str,
        dest_dir: str | Path,
        *,
        chunk_bytes: int = 8 * 1024 * 1024,
        overwrite: bool = False,
    ) -> Path:
        """Stream a product zip from the zipper endpoint to ``dest_dir``.

        Written to a ``.part`` file and renamed on completion, so an interrupted
        download can never be mistaken for a finished product.
        """
        product_id = product.id if isinstance(product, Product) else product
        name = product.name if isinstance(product, Product) else product_id
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / f"{name}.zip"
        if target.exists() and not overwrite:
            log.info("product already downloaded: %s", target)
            return target

        part = target.with_suffix(".zip.part")
        url = f"{ZIPPER_URL}/Products({product_id})/$value"
        log.info("downloading %s", name)
        with self._client.stream("GET", url, headers=self._auth_headers()) as response:
            if response.status_code != 200:
                raise CDSEError(
                    f"CDSE download failed ({response.status_code}) for {name}"
                )
            with part.open("wb") as handle:
                for chunk in response.iter_bytes(chunk_bytes):
                    handle.write(chunk)
        part.rename(target)
        log.info("downloaded %s (%.1f MB)", target.name, target.stat().st_size / 1e6)
        return target

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> CDSEClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def recent_window(days: int = 7, *, now: datetime | None = None) -> tuple[datetime, datetime]:
    """Convenience window for "the last N days", UTC-aware."""
    end = now or datetime.now(UTC)
    return end - timedelta(days=days), end


def iter_products(products: Iterator[Product], platforms: Sequence[str]) -> Iterator[Product]:
    wanted = {p.upper() for p in platforms}
    return (p for p in products if (p.platform or "") in wanted)
