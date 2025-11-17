from pathlib import Path
from typing import List
import wfdb

from .config import DATA_DIR


def download_mitbih(data_dir: Path | None = None, records: List[str] | None = None) -> List[str]:
    """
    Download the MIT-BIH Arrhythmia Database to the local cache using wfdb.

    Args:
        data_dir: Destination directory (default: config.DATA_DIR)
        records: Optional list of record names to download (e.g., ["100", "101"]).
                 If None, downloads the full database index and fetches all records.

    Returns:
        List of record names available locally.
    """
    dest = Path(data_dir) if data_dir is not None else DATA_DIR
    dest.mkdir(parents=True, exist_ok=True)

    # Download record list
    # wfdb supports dl_database; if records not provided, get index
    if records is None:
        try:
            # Fetch the list of records from PhysioNet (mitdb/RECORDS)
            idx_path = dest / "RECORDS"
            if not idx_path.exists():
                wfdb.dl_database("mitdb", dest_dir=str(dest), dl_files=["RECORDS"])
            with open(idx_path, "r", encoding="utf-8") as f:
                records = [line.strip() for line in f if line.strip()]
        except Exception:
            # Fallback to a common subset if index download failed
            records = [
                "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
                "111", "112", "113", "114", "115", "116", "117", "118", "119",
                "121", "122", "123", "124", "200", "201", "202", "203", "205",
                "207", "208", "209", "210", "212", "213", "214", "215", "217",
                "219", "220", "221", "222", "223", "228", "230", "231", "232", "233", "234",
            ]

    # For each record, download signal and annotation files
    for rec in records:
        try:
            wfdb.dl_database(
                "mitdb",
                dest_dir=str(dest),
                dl_files=[f"{rec}.dat", f"{rec}.hea", f"{rec}.atr"],
                keep_subdirs=False,
            )
        except Exception:
            # If batch download fails, try individual files silently
            for ext in ("dat", "hea", "atr"):
                try:
                    wfdb.dl_database("mitdb", dest_dir=str(dest), dl_files=[f"{rec}.{ext}"], keep_subdirs=False)
                except Exception:
                    pass

    # Verify availability
    local_records: List[str] = []
    for rec in records:
        if (dest / f"{rec}.hea").exists():
            local_records.append(rec)

    return local_records
