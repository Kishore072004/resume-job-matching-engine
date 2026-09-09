import os
import sys
from pathlib import Path
import pandas as pd

# Reconfigure stdout/stderr to handle UTF-8 safely on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def get_source(file_path: Path, raw_dir: Path) -> str:
    """Extract source name as the top-level directory under data/raw/."""
    try:
        rel_path = file_path.relative_to(raw_dir)
        return rel_path.parts[0] if len(rel_path.parts) > 1 else rel_path.stem
    except Exception:
        return file_path.parent.name


def load_dataframe(file_path: Path, ext: str) -> pd.DataFrame:
    """Load dataframe with graceful encoding fallback handling."""
    if ext in [".csv", ".tsv"]:
        sep = "\t" if ext == ".tsv" else ","
        encodings = ["utf-8", "utf-8-sig", "latin1", "iso-8859-1", "cp1252"]
        for enc in encodings:
            try:
                return pd.read_csv(file_path, sep=sep, encoding=enc, low_memory=False)
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        # Fallback with encoding_errors replacement
        return pd.read_csv(
            file_path,
            sep=sep,
            encoding="utf-8",
            encoding_errors="replace",
            low_memory=False,
        )
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    elif ext == ".parquet":
        return pd.read_parquet(file_path)
    elif ext in [".json", ".jsonl"]:
        try:
            return pd.read_json(file_path)
        except Exception:
            return pd.read_json(file_path, lines=True)
    else:
        raise ValueError(f"Unsupported extension: {ext}")


def inspect_datasets():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    raw_dir = project_root / "data" / "raw"
    inventory_file = project_root / "data" / "dataset_inventory.csv"

    supported_extensions = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".jsonl", ".parquet"}

    if not raw_dir.exists():
        print(f"Error: Directory '{raw_dir}' does not exist.")
        sys.exit(1)

    discovered_files = sorted(
        [p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in supported_extensions]
    )

    print(f"Discovered {len(discovered_files)} supported raw dataset files in '{raw_dir}'.\n")

    inventory_records = []

    for file_path in discovered_files:
        file_name = file_path.name
        rel_file_path = file_path.relative_to(project_root).as_posix()
        file_type = file_path.suffix.lstrip(".").lower()
        source = get_source(file_path, raw_dir)

        print("=" * 80)
        print(f"File Name    : {file_name}")
        print(f"File Path    : {rel_file_path}")
        print(f"File Type    : {file_type.upper()}")
        print(f"Source       : {source}")

        try:
            df = load_dataframe(file_path, file_path.suffix.lower())
            rows = len(df)
            cols = len(df.columns)
            dup_rows = int(df.duplicated().sum())

            print(f"Number of Rows    : {rows}")
            print(f"Number of Columns : {cols}")
            print(f"Duplicate Rows    : {dup_rows}")
            print(f"Column Names      : {list(df.columns)}")

            print("\nMissing Values Count Per Column:")
            missing_counts = df.isnull().sum()
            for col_name, count in missing_counts.items():
                pct = (count / rows * 100) if rows > 0 else 0
                print(f"  - {col_name}: {count} ({pct:.2f}%)")

            print("\nFirst 3 Rows:")
            sample_df = df.head(3).copy()
            # Truncate extremely long string representations for clean display
            for col in sample_df.columns:
                sample_df[col] = sample_df[col].astype(str).apply(
                    lambda s: (s[:100] + "...") if len(s) > 100 else s
                )
            print(sample_df.to_string(index=False))

            inventory_records.append({
                "source": source,
                "file_name": file_name,
                "file_path": rel_file_path,
                "file_type": file_type,
                "rows": rows,
                "columns": cols,
                "duplicate_rows": dup_rows,
            })

        except Exception as e:
            print(f"ERROR inspecting {file_name}: {e}")
            inventory_records.append({
                "source": source,
                "file_name": file_name,
                "file_path": rel_file_path,
                "file_type": file_type,
                "rows": -1,
                "columns": -1,
                "duplicate_rows": -1,
            })

        print("=" * 80 + "\n")

    # Create dataset_inventory.csv
    inventory_df = pd.DataFrame(inventory_records)
    inventory_file.parent.mkdir(parents=True, exist_ok=True)
    inventory_df.to_csv(inventory_file, index=False)
    print(f"Saved dataset inventory to: {inventory_file.relative_to(project_root).as_posix()}")


if __name__ == "__main__":
    inspect_datasets()
