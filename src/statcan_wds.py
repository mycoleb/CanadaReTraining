from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


WDS_BASE = "https://www150.statcan.gc.ca/t1/wds/en"


@dataclass
class FullTableDownload:
    cube: str
    url: str


def _post_json(endpoint: str, payload: dict, timeout: int = 60) -> dict:
    r = requests.post(endpoint, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_full_table_download_csv_url(cube: str) -> str:
    """
    Get a URL to download the *entire* table as a ZIPped CSV via StatsCan WDS.

    cube examples:
      - "1410043201" (Table 14-10-0432-01)
      - "1410028701" (Table 14-10-0287-01)
    """
    endpoint = f"{WDS_BASE}/getFullTableDownloadCSV"
    payload = {"productId": cube, "downloadType": "csv"}
    data = _post_json(endpoint, payload)

    # WDS responses are usually like:
    # {"status":"SUCCESS","object":"https://.../file.zip", ...}
    if "object" not in data or not isinstance(data["object"], str):
        raise RuntimeError(f"Unexpected WDS response for cube={cube}: {data}")

    return data["object"]


def download_and_extract_first_csv(zip_url: str, cache_dir: Path) -> Path:
    """
    Downloads a ZIP and extracts the first CSV found. Returns extracted CSV path.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    zip_path = cache_dir / "table.zip"
    r = requests.get(zip_url, timeout=120)
    r.raise_for_status()
    zip_path.write_bytes(r.content)

    with zipfile.ZipFile(zip_path, "r") as z:
        csv_members = [m for m in z.namelist() if m.lower().endswith(".csv")]
        if not csv_members:
            raise RuntimeError(f"No CSV found in ZIP from {zip_url}")
        member = csv_members[0]
        out_path = cache_dir / Path(member).name
        with z.open(member) as f:
            out_path.write_bytes(f.read())

    return out_path


def load_statcan_cube_full_table(cube: str, data_dir: Path) -> pd.DataFrame:
    """
    Downloads and loads a full StatsCan table into a DataFrame.
    """
    cube_dir = data_dir / f"statcan_{cube}"
    cube_dir.mkdir(parents=True, exist_ok=True)

    cached = list(cube_dir.glob("*.csv"))
    if cached:
        return pd.read_csv(cached[0])

    url = get_full_table_download_csv_url(cube)
    csv_path = download_and_extract_first_csv(url, cube_dir)
    return pd.read_csv(csv_path)
