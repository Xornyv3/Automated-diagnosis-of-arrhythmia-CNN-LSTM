from __future__ import annotations
from pathlib import Path
from typing import List
import wfdb

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "mitdb"


def download_mitbih(dest: Path | str | None = None, records: List[str] | None = None) -> List[str]:
    """Download selected MIT-BIH records via WFDB into dest and return the list of available records.

    If records is None, uses the RECORDS index or a default set.
    """
    dest_path = Path(dest) if dest is not None else DEFAULT_DATA_DIR
    dest_path.mkdir(parents=True, exist_ok=True)

    if records is None:
        try:
            records = wfdb.get_record_list("mitdb")
        except Exception:
            records = [
                "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
                "111", "112", "113", "114", "115", "116", "117", "118", "119",
                "121", "122", "123", "124", "200", "201", "202", "203", "205",
                "207", "208", "209", "210", "212", "213", "214", "215", "217",
                "219", "220", "221", "222", "223", "228", "230", "231", "232", "233", "234",
            ]

    for rec in records:
        try:
            wfdb.dl_database(
                "mitdb",
                dl_dir=str(dest_path),
                records=[rec],
                annotators=["atr"],
                keep_subdirs=False,
            )
        except Exception:
            for ext in ("dat", "hea", "atr"):
                try:
                    wfdb.dl_files(
                        "mitdb",
                        dl_dir=str(dest_path),
                        files=[f"{rec}.{ext}"],
                        keep_subdirs=False,
                    )
                except Exception:
                    pass

    local = [rec for rec in records if (dest_path / f"{rec}.hea").exists()]
    return local
