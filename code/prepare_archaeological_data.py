"""Create the archaeological analysis table from the bundled upstream data.

The transformation is intentionally small and explicit:
1. retain archaeological observations (``Arch == 1``);
2. retain records coded as a single manufacturing chain;
3. count the binary ``PU.*`` operation columns;
4. use the midpoint of the reported age interval; and
5. flag sequences above the conservative non-cumulative comparison maximum
   of six procedural units.

The source CSV is encoded as Windows-1252.  The output is UTF-8.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "procedural_units_obfus.csv"
DEFAULT_OUTPUT = ROOT / "data" / "archaeological_single_chain_data.csv"
BASELINE_PROCEDURAL_UNITS = 6


def build_analysis_table(source: Path) -> pd.DataFrame:
    df = pd.read_csv(source, encoding="cp1252")
    pu_columns = [column for column in df.columns if column.startswith("PU.")]
    if not pu_columns:
        raise ValueError("No procedural-unit columns with prefix 'PU.' were found")

    arch = pd.to_numeric(df["Arch"], errors="coerce").eq(1)
    single_chain = (
        df["Single.Chain"]
        .astype("string")
        .str.strip()
        .str.casefold()
        .eq("yes")
    )
    out = df.loc[arch & single_chain].copy()

    operations = out[pu_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    out["procedural_units"] = operations.sum(axis=1).astype(int)
    out["age_mid_ka"] = (
        pd.to_numeric(out["KA.young"], errors="raise")
        + pd.to_numeric(out["KA.old"], errors="raise")
    ) / 2.0
    out["above_noncumulative_baseline"] = (
        out["procedural_units"] > BASELINE_PROCEDURAL_UNITS
    )
    return out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    table = build_analysis_table(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False, encoding="utf-8")
    print(
        f"Wrote {len(table)} dated archaeological single-chain records to "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
