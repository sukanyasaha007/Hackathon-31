"""
Download HTS tariff data and CROSS rulings for the Tariff Classification Agent.

Data sources:
1. HTS Annual Tariff CSV from USITC DataWeb
2. CROSS Rulings bulk download (NY + HQ) from CBP

Usage: python3 -m src.data.download
"""

import os
import sys
import zipfile
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import HTS_DIR, CROSS_DIR


def download_file(url: str, dest: Path, description: str, params: dict = None):
    """Download a file with progress indication."""
    print(f"Downloading {description}...")
    print(f"  URL: {url}")
    with httpx.stream("GET", url, params=params, follow_redirects=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded / 1e6:.1f} MB / {total / 1e6:.1f} MB ({pct:.0f}%)", end="", flush=True)
                else:
                    print(f"\r  {downloaded / 1e6:.1f} MB", end="", flush=True)
    print(f"\n  Saved to {dest}")


def download_hts_data():
    """Download HTS chapter PDFs for key sections (electronics, vehicles, instruments)."""
    chapters = [84, 85, 87, 90, 91, 95, 96]  # Electronics, vehicles, instruments, misc
    base_url = "https://hts.usitc.gov/reststop/file"

    for ch in chapters:
        filename = f"Chapter {ch}"
        dest = HTS_DIR / f"chapter_{ch}.pdf"
        if dest.exists():
            print(f"  Chapter {ch} already downloaded, skipping.")
            continue
        try:
            download_file(
                base_url,
                dest,
                f"HTS Chapter {ch}",
                params={"release": "currentRelease", "filename": filename},
            )
        except Exception as e:
            print(f"  Error downloading Chapter {ch}: {e}")

    # Also download General Rules of Interpretation
    gri_dest = HTS_DIR / "general_rules_of_interpretation.pdf"
    if not gri_dest.exists():
        try:
            download_file(
                base_url,
                gri_dest,
                "General Rules of Interpretation",
                params={"release": "currentRelease", "filename": "General Rules of Interpretation"},
            )
        except Exception as e:
            print(f"  Error downloading GRI: {e}")

    # Download General Notes
    gn_dest = HTS_DIR / "general_notes.pdf"
    if not gn_dest.exists():
        try:
            download_file(
                base_url,
                gn_dest,
                "General Notes",
                params={"release": "currentRelease", "filename": "General Notes"},
            )
        except Exception as e:
            print(f"  Error downloading General Notes: {e}")


def download_cross_rulings():
    """Download CROSS rulings bulk data (NY + HQ collections)."""
    base_url = "https://rulings.cbp.gov/api/downloadLatestDocuments"

    # Check last update
    try:
        r = httpx.get("https://rulings.cbp.gov/api/stat/lastupdate", timeout=15)
        stats = r.json()
        print(f"CROSS database stats:")
        print(f"  Last updated: {stats['lastUpdateDate']}")
        print(f"  Total rulings: {stats['totalSearchableRulingsCount']}")
        print(f"  New rulings added: {stats['numberOfNewRlingsAdded']}")
    except Exception as e:
        print(f"Could not fetch CROSS stats: {e}")

    for collection in ["NY", "HQ"]:
        dest = CROSS_DIR / f"cross_{collection.lower()}_latest.zip"
        extracted_dir = CROSS_DIR / collection.lower()

        print(f"\nDownloading CROSS {collection} rulings...")
        try:
            download_file(
                base_url,
                dest,
                f"CROSS {collection} rulings (ZIP)",
                params={"collection": collection},
            )

            # Extract ZIP
            extracted_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Extracting to {extracted_dir}...")
            with zipfile.ZipFile(dest, "r") as zf:
                zf.extractall(extracted_dir)
                print(f"  Extracted {len(zf.namelist())} files")

        except Exception as e:
            print(f"  Error downloading CROSS {collection}: {e}")


def main():
    print("=" * 60)
    print("TARIFF DATA DOWNLOAD")
    print("=" * 60)

    print("\n--- HTS Tariff Schedule ---")
    download_hts_data()

    print("\n--- CROSS Rulings ---")
    download_cross_rulings()

    print("\n--- Summary ---")
    hts_files = list(HTS_DIR.glob("*"))
    cross_files = list(CROSS_DIR.rglob("*"))
    print(f"HTS files: {len(hts_files)}")
    print(f"CROSS files: {len(cross_files)}")

    # Total size
    total = sum(f.stat().st_size for f in HTS_DIR.rglob("*") if f.is_file())
    total += sum(f.stat().st_size for f in CROSS_DIR.rglob("*") if f.is_file())
    print(f"Total data size: {total / 1e6:.1f} MB")
    print("Done.")


if __name__ == "__main__":
    main()
