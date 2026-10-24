"""Convert the raw H&M CSVs to compact Parquet, and load them back.

Usage:
    python -m src.data_loader                   # full conversion -> data/processed/
    python -m src.data_loader --sample-frac 0.05  # 5% of customers -> data/processed/sample_0.05/

Customers get a dense int32 index (`customer_idx`) that replaces the 64-char hex
`customer_id` everywhere except customers.parquet, which keeps the mapping.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

RAW_DIR = Path("data/raw")
# DATA_DIR lets Azure ML / Colab / Kaggle point at wherever the processed Parquet is mounted.
PROCESSED_DIR = Path(os.environ.get("DATA_DIR", "data/processed"))

# Free-text article columns that should stay plain strings (too many unique values for `category`).
ARTICLE_TEXT_COLS = ["prod_name", "detail_desc"]


def convert_customers(raw_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "customers.csv", dtype={"customer_id": str})
    df.insert(0, "customer_idx", np.arange(len(df), dtype=np.int32))
    # FN / Active are 1.0 or NaN in the raw file; store as 0/1 flags.
    df["FN"] = df["FN"].fillna(0).astype(np.int8)
    df["Active"] = df["Active"].fillna(0).astype(np.int8)
    df["age"] = df["age"].astype(np.float32)  # keeps NaN for unknown age
    for col in ["club_member_status", "fashion_news_frequency", "postal_code"]:
        df[col] = df[col].astype("category")
    return df


def convert_articles(raw_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "articles.csv")
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif col not in ARTICLE_TEXT_COLS:
            df[col] = df[col].astype("category")
    df["article_id"] = df["article_id"].astype(np.int32)
    return df


def convert_transactions(raw_dir: Path, customer_ids: pd.Index) -> pd.DataFrame:
    """Stream the 3.5 GB CSV in blocks so the full string column is never in memory."""
    reader = pacsv.open_csv(
        raw_dir / "transactions_train.csv",
        read_options=pacsv.ReadOptions(block_size=128 << 20),
        convert_options=pacsv.ConvertOptions(
            column_types={
                "t_dat": pa.date32(),
                "customer_id": pa.string(),
                "article_id": pa.int32(),
                "price": pa.float32(),
                "sales_channel_id": pa.int8(),
            }
        ),
    )
    parts = []
    for batch in reader:
        # Look up each unique customer string once per block, then broadcast via the codes.
        encoded = pc.dictionary_encode(batch.column("customer_id"))
        uniq_idx = customer_ids.get_indexer(encoded.dictionary.to_numpy(zero_copy_only=False))
        if (uniq_idx < 0).any():
            raise ValueError("transactions contain customer_ids missing from customers.csv")
        cust_idx = uniq_idx.astype(np.int32)[encoded.indices.to_numpy()]
        parts.append(
            pd.DataFrame(
                {
                    "t_dat": batch.column("t_dat").to_numpy(zero_copy_only=False).astype("datetime64[s]"),
                    "customer_idx": cust_idx,
                    "article_id": batch.column("article_id").to_numpy(),
                    "price": batch.column("price").to_numpy(),
                    "sales_channel_id": batch.column("sales_channel_id").to_numpy(),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def sample_customers(customers: pd.DataFrame, frac: float, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(len(customers) * frac)
    return np.sort(rng.choice(customers["customer_idx"].to_numpy(), size=n, replace=False))


def output_dir(sample_frac: float | None) -> Path:
    return PROCESSED_DIR if sample_frac is None else PROCESSED_DIR / f"sample_{sample_frac}"


def convert(raw_dir: Path = RAW_DIR, sample_frac: float | None = None) -> Path:
    out = output_dir(sample_frac)
    out.mkdir(parents=True, exist_ok=True)

    customers = convert_customers(raw_dir)
    articles = convert_articles(raw_dir)
    transactions = convert_transactions(raw_dir, pd.Index(customers["customer_id"]))

    if sample_frac is not None:
        keep = sample_customers(customers, sample_frac)
        customers = customers[customers["customer_idx"].isin(keep)].reset_index(drop=True)
        for col in customers.select_dtypes("category"):
            customers[col] = customers[col].cat.remove_unused_categories()
        transactions = transactions[transactions["customer_idx"].isin(keep)].reset_index(drop=True)

    customers.to_parquet(out / "customers.parquet", index=False)
    articles.to_parquet(out / "articles.parquet", index=False)
    transactions.to_parquet(out / "transactions.parquet", index=False, row_group_size=1_000_000)
    return out


def load(
    name: str,
    sample_frac: float | None = None,
    columns: list[str] | None = None,
    filters: list[tuple] | None = None,
) -> pd.DataFrame:
    """Load one processed table: 'transactions', 'customers' or 'articles'.

    `filters` are pyarrow row filters, e.g. [("t_dat", ">=", date)]; row groups outside them are skipped on read.
    """
    table = pq.read_table(output_dir(sample_frac) / f"{name}.parquet", columns=columns, filters=filters)
    # self_destruct frees each Arrow column as it is converted, so peak memory stays near one copy.
    return table.to_pandas(split_blocks=True, self_destruct=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--sample-frac", type=float, default=None)
    args = parser.parse_args()
    print(f"wrote {convert(args.raw_dir, args.sample_frac)}")
