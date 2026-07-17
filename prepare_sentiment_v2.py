# -*- coding: utf-8 -*-
"""Prepare reproducible v2 sentiment datasets without changing legacy data.

Outputs use the common schema ``text,label,movie,row_id,source``.  Canonical
reviews are frozen into deterministic 70/15/15 splits, while the external
CC0 DMSC sample is kept in a separate training-only file.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

from feature_engineering import (
    COL_MOVIE,
    COL_SCORE,
    COL_TEXT,
    COL_TIME,
    COL_USER,
    clean_comment_text,
    clean_structured_data,
    engineer_features,
    load_raw_data,
)


ROOT = Path(__file__).resolve().parent
CANONICAL_INPUT = ROOT / "assets" / "data" / "douban_comments.csv"
EXTERNAL_INPUT = ROOT / "assets" / "data" / "external" / "DMSC.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "sentiment" / "v2"

OUTPUT_COLUMNS = ["text", "label", "movie", "row_id", "source"]
LABEL_ORDER = ["负向", "中性", "正向"]
CANONICAL_SOURCE = "canonical_douban_2025_2026"
EXTERNAL_SOURCE = "kaggle_utmhikari_dmsc"
EXTERNAL_SOURCE_URL = (
    "https://www.kaggle.com/datasets/utmhikari/"
    "doubanmovieshortcomments?select=DMSC.csv"
)
EXTERNAL_LICENSE = {
    "name": "CC0 1.0 Universal (Public Domain Dedication)",
    "spdx": "CC0-1.0",
    "url": "https://creativecommons.org/publicdomain/zero/1.0/",
}

DMSC_COLUMNS = [
    "ID",
    "Movie_Name_EN",
    "Movie_Name_CN",
    "Date",
    "Star",
    "Comment",
]
EXTERNAL_CHUNK_SIZE = 100_000
INFORMATION_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare frozen canonical splits and a DMSC training-only sample."
    )
    parser.add_argument(
        "--external-per-class",
        type=int,
        default=50_000,
        help="Deterministically sample this many DMSC rows for each sentiment class.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split and sampling seed.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for data/, splits/, and manifest.json.",
    )
    args = parser.parse_args()
    if args.external_per_class <= 0:
        parser.error("--external-per-class must be greater than zero")
    return args


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_row_id(source: str, *values: object) -> str:
    payload = json.dumps(
        [source, *("" if pd.isna(value) else str(value).strip() for value in values)],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def text_fingerprint(text: str) -> bytes:
    """Compact collision-resistant key used for overlap and duplicate checks."""

    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def is_informative(text: object) -> bool:
    value = "" if pd.isna(text) else str(text).strip()
    # Two semantic characters remove blank, punctuation-only, and one-symbol rows.
    return len(INFORMATION_RE.findall(value)) >= 2


def label_from_star(star: int) -> str:
    if star <= 2:
        return "负向"
    if star == 3:
        return "中性"
    return "正向"


def class_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    return {label: int(counts[label]) for label in LABEL_ORDER}


def prepare_canonical() -> tuple[pd.DataFrame, set[bytes], dict[str, object]]:
    """Rebuild canonical records with the project's established cleaning code."""

    raw = load_raw_data(CANONICAL_INPUT)
    structured, cleaning_summary = clean_structured_data(raw)
    featured, feature_summary = engineer_features(structured)

    all_clean_texts = featured["清洗评论"].map(clean_comment_text)
    canonical_overlap_keys = {
        text_fingerprint(text)
        for text in all_clean_texts.astype(str)
        if str(text).strip()
    }

    row_ids = [
        stable_row_id(
            CANONICAL_SOURCE,
            row[COL_MOVIE],
            row[COL_USER],
            row[COL_SCORE],
            row.get("评分_5分制", ""),
            row[COL_TIME],
            row[COL_TEXT],
        )
        for _, row in featured.iterrows()
    ]
    canonical = pd.DataFrame(
        {
            "text": all_clean_texts.astype(str),
            "label": featured["情感标签"].astype(str),
            "movie": featured[COL_MOVIE].astype(str),
            "row_id": row_ids,
            "source": CANONICAL_SOURCE,
        }
    )

    before_low_info = len(canonical)
    canonical = canonical.loc[canonical["text"].map(is_informative)].copy()
    low_info_removed = before_low_info - len(canonical)

    canonical = canonical.sort_values("row_id", kind="stable").reset_index(drop=True)
    duplicate_row_ids = int(canonical.duplicated("row_id").sum())
    if duplicate_row_ids:
        raise AssertionError(f"canonical row_id collision/duplicate count: {duplicate_row_ids}")

    before_text_dedup = len(canonical)
    canonical = canonical.drop_duplicates("text", keep="first").reset_index(drop=True)
    duplicate_text_removed = before_text_dedup - len(canonical)
    canonical = canonical[OUTPUT_COLUMNS]

    if not set(canonical["label"]).issubset(set(LABEL_ORDER)):
        raise AssertionError("canonical data contains an unknown label")
    if canonical["row_id"].duplicated().any() or canonical["text"].duplicated().any():
        raise AssertionError("canonical data is not unique by row_id and text")

    summary = {
        "raw_rows": int(len(raw)),
        "structured_rows": int(len(structured)),
        "featured_rows": int(len(featured)),
        "canonical_overlap_universe_texts": int(len(canonical_overlap_keys)),
        "low_information_rows_removed": int(low_info_removed),
        "duplicate_clean_text_rows_removed": int(duplicate_text_removed),
        "output_rows": int(len(canonical)),
        "class_counts": class_counts(canonical),
        "existing_cleaning_summary": cleaning_summary,
        "existing_feature_summary": feature_summary,
    }
    return canonical, canonical_overlap_keys, summary


def freeze_splits(
    canonical: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = canonical.sort_values("row_id", kind="stable").reset_index(drop=True)
    train, remainder = train_test_split(
        ordered,
        test_size=0.30,
        random_state=seed,
        stratify=ordered["label"],
    )
    valid, test = train_test_split(
        remainder,
        test_size=0.50,
        random_state=seed,
        stratify=remainder["label"],
    )
    return tuple(
        frame.sort_values("row_id", kind="stable").reset_index(drop=True)[OUTPUT_COLUMNS]
        for frame in (train, valid, test)
    )


def sampling_rank(seed: int, row_id: str) -> int:
    payload = f"{seed}\x1f{row_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def add_to_bounded_sample(
    heap: list[tuple[int, str, str, str, str, str]],
    limit: int,
    rank: int,
    row_id: str,
    text: str,
    label: str,
    movie: str,
) -> None:
    # Negated rank makes heap[0] the currently worst (largest) retained rank.
    item = (-rank, row_id, text, label, movie, EXTERNAL_SOURCE)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif rank < -heap[0][0]:
        heapq.heapreplace(heap, item)


def prepare_external(
    canonical_overlap_keys: set[bytes],
    per_class: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Stream, clean, deduplicate, and hash-sample the 400 MB DMSC file."""

    heaps: dict[str, list[tuple[int, str, str, str, str, str]]] = {
        label: [] for label in LABEL_ORDER
    }
    seen_external_texts: set[bytes] = set()
    eligible_counts: Counter[str] = Counter()
    scan = Counter()

    reader = pd.read_csv(
        EXTERNAL_INPUT,
        encoding="utf-8-sig",
        usecols=DMSC_COLUMNS,
        dtype=str,
        keep_default_na=False,
        chunksize=EXTERNAL_CHUNK_SIZE,
    )
    for chunk_index, chunk in enumerate(reader, start=1):
        chunk = chunk[DMSC_COLUMNS]
        scan["raw_rows"] += len(chunk)
        cleaned_texts = chunk["Comment"].map(clean_comment_text)
        for values, text in zip(chunk.itertuples(index=False, name=None), cleaned_texts):
            external_id, movie_en, movie_cn, review_date, star_raw, _comment = values
            text = str(text).strip()
            if not text:
                scan["empty_text_rows_removed"] += 1
                continue
            if not is_informative(text):
                scan["low_information_rows_removed"] += 1
                continue

            try:
                star_float = float(str(star_raw).strip())
                star = int(star_float)
            except (TypeError, ValueError, OverflowError):
                scan["invalid_star_rows_removed"] += 1
                continue
            if star_float != star or star not in {1, 2, 3, 4, 5}:
                scan["invalid_star_rows_removed"] += 1
                continue

            fingerprint = text_fingerprint(text)
            if fingerprint in canonical_overlap_keys:
                scan["canonical_overlap_rows_removed"] += 1
                continue
            if fingerprint in seen_external_texts:
                scan["duplicate_clean_text_rows_removed"] += 1
                continue
            seen_external_texts.add(fingerprint)

            label = label_from_star(star)
            eligible_counts[label] += 1
            movie = str(movie_cn).strip() or str(movie_en).strip()
            row_id = stable_row_id(
                EXTERNAL_SOURCE,
                external_id,
                movie,
                review_date,
                star,
                text,
            )
            rank = sampling_rank(seed, row_id)
            add_to_bounded_sample(
                heaps[label], per_class, rank, row_id, text, label, movie
            )

        if chunk_index % 5 == 0:
            print(
                f"DMSC scanned {scan['raw_rows']:,} rows; "
                f"eligible={sum(eligible_counts.values()):,}",
                flush=True,
            )

    insufficient = {
        label: int(eligible_counts[label])
        for label in LABEL_ORDER
        if eligible_counts[label] < per_class
    }
    if insufficient:
        raise AssertionError(
            f"DMSC does not contain {per_class} eligible unique rows per class: {insufficient}"
        )

    sampled_rows: list[dict[str, str]] = []
    for label in LABEL_ORDER:
        for _neg_rank, row_id, text, item_label, movie, source in heaps[label]:
            sampled_rows.append(
                {
                    "text": text,
                    "label": item_label,
                    "movie": movie,
                    "row_id": row_id,
                    "source": source,
                }
            )
    external = pd.DataFrame(sampled_rows, columns=OUTPUT_COLUMNS)
    external = external.sort_values(["label", "row_id"], kind="stable").reset_index(drop=True)

    if external["row_id"].duplicated().any() or external["text"].duplicated().any():
        raise AssertionError("sampled external data is not unique by row_id and text")
    expected_counts = {label: per_class for label in LABEL_ORDER}
    if class_counts(external) != expected_counts:
        raise AssertionError(
            f"external sampled class counts differ from requested counts: {class_counts(external)}"
        )

    summary = {
        "raw_rows": int(scan["raw_rows"]),
        "empty_text_rows_removed": int(scan["empty_text_rows_removed"]),
        "low_information_rows_removed": int(scan["low_information_rows_removed"]),
        "invalid_star_rows_removed": int(scan["invalid_star_rows_removed"]),
        "canonical_overlap_rows_removed": int(scan["canonical_overlap_rows_removed"]),
        "duplicate_clean_text_rows_removed": int(
            scan["duplicate_clean_text_rows_removed"]
        ),
        "eligible_unique_class_counts": {
            label: int(eligible_counts[label]) for label in LABEL_ORDER
        },
        "sampled_rows": int(len(external)),
        "sampled_class_counts": class_counts(external),
        "sampling_method": "smallest SHA256(seed + row_id) rank per class",
    }
    return external, summary


def intersection_count(left: pd.DataFrame, right: pd.DataFrame, column: str) -> int:
    return len(set(left[column]) & set(right[column]))


def validate_outputs(
    canonical: pd.DataFrame,
    canonical_overlap_keys: set[bytes],
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    external: pd.DataFrame,
) -> dict[str, object]:
    split_frames = {"train": train, "valid": valid, "test": test}
    for name, frame in {"canonical": canonical, **split_frames, "external": external}.items():
        if list(frame.columns) != OUTPUT_COLUMNS:
            raise AssertionError(f"{name} columns do not match {OUTPUT_COLUMNS}")
        if frame.isna().any().any():
            raise AssertionError(f"{name} contains null values")
        if frame["text"].astype(str).str.strip().eq("").any():
            raise AssertionError(f"{name} contains blank text")
        if frame["row_id"].duplicated().any():
            raise AssertionError(f"{name} contains duplicate row_id values")
        if frame["text"].duplicated().any():
            raise AssertionError(f"{name} contains duplicate text values")

    pair_names = [("train", "valid"), ("train", "test"), ("valid", "test")]
    row_id_intersections = {
        f"{left}_{right}": intersection_count(
            split_frames[left], split_frames[right], "row_id"
        )
        for left, right in pair_names
    }
    text_intersections = {
        f"{left}_{right}": intersection_count(
            split_frames[left], split_frames[right], "text"
        )
        for left, right in pair_names
    }
    external_overlap = {
        "all_canonical_original_text": len(
            {text_fingerprint(text) for text in external["text"]}
            & canonical_overlap_keys
        ),
        "canonical_text": intersection_count(external, canonical, "text"),
        "canonical_row_id": intersection_count(external, canonical, "row_id"),
        "train_text": intersection_count(external, train, "text"),
        "valid_text": intersection_count(external, valid, "text"),
        "test_text": intersection_count(external, test, "text"),
    }

    if sum(row_id_intersections.values()) != 0:
        raise AssertionError(f"frozen split row_id overlap: {row_id_intersections}")
    if sum(text_intersections.values()) != 0:
        raise AssertionError(f"frozen split text overlap: {text_intersections}")
    if sum(external_overlap.values()) != 0:
        raise AssertionError(f"external/canonical overlap: {external_overlap}")
    if len(train) + len(valid) + len(test) != len(canonical):
        raise AssertionError("frozen split row counts do not reconstruct canonical data")

    actual_ratios = {
        "train": len(train) / len(canonical),
        "valid": len(valid) / len(canonical),
        "test": len(test) / len(canonical),
    }
    return {
        "passed": True,
        "split_row_id_intersections": row_id_intersections,
        "split_text_intersections": text_intersections,
        "external_overlap": external_overlap,
        "split_actual_ratios": {
            name: round(value, 6) for name, value in actual_ratios.items()
        },
        "canonical_reconstructed_by_splits": True,
        "all_frames_unique_by_row_id_and_text": True,
        "external_is_training_only": True,
    }


def csv_output_metadata(path: Path, frame: pd.DataFrame) -> dict[str, object]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "rows": int(len(frame)),
        "class_counts": class_counts(frame),
        "columns": OUTPUT_COLUMNS,
        "encoding": "utf-8-sig",
    }


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def build_manifest(
    output_root: Path,
    seed: int,
    external_per_class: int,
    canonical_summary: dict[str, object],
    external_summary: dict[str, object],
    validation: dict[str, object],
    outputs: Iterable[tuple[Path, pd.DataFrame]],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "pipeline": "prepare_sentiment_v2.py",
        "parameters": {
            "seed": seed,
            "external_per_class": external_per_class,
            "canonical_split": [0.70, 0.15, 0.15],
            "external_policy": "training-only; never included in canonical valid/test",
            "low_information_policy": (
                "after project text cleaning, require at least two Chinese/Latin/digit characters"
            ),
        },
        "sources": {
            "canonical": {
                "name": "project canonical Douban comments",
                "source_url": "https://movie.douban.com/",
                "input_path": CANONICAL_INPUT.relative_to(ROOT).as_posix(),
                "input_sha256": sha256_file(CANONICAL_INPUT),
            },
            "external": {
                "name": "Douban Movie Short Comments Dataset",
                "source_url": EXTERNAL_SOURCE_URL,
                "license": EXTERNAL_LICENSE,
                "input_path": EXTERNAL_INPUT.relative_to(ROOT).as_posix(),
                "input_sha256": sha256_file(EXTERNAL_INPUT),
            },
        },
        "canonical_processing": canonical_summary,
        "external_processing": external_summary,
        "outputs": {
            path.relative_to(output_root).as_posix(): csv_output_metadata(path, frame)
            for path, frame in outputs
        },
        "overlap_checks": validation,
    }


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    data_dir = output_root / "data"
    split_dir = output_root / "splits"
    data_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing canonical data...", flush=True)
    canonical, canonical_overlap_keys, canonical_summary = prepare_canonical()
    train, valid, test = freeze_splits(canonical, args.seed)

    print("Preparing external DMSC training data...", flush=True)
    external, external_summary = prepare_external(
        canonical_overlap_keys,
        per_class=args.external_per_class,
        seed=args.seed,
    )

    validation = validate_outputs(
        canonical, canonical_overlap_keys, train, valid, test, external
    )

    canonical_path = data_dir / "canonical.csv"
    external_path = data_dir / "external_train.csv"
    train_path = split_dir / "train.csv"
    valid_path = split_dir / "valid.csv"
    test_path = split_dir / "test.csv"
    output_frames = [
        (canonical_path, canonical),
        (external_path, external),
        (train_path, train),
        (valid_path, valid),
        (test_path, test),
    ]
    for path, frame in output_frames:
        write_csv(path, frame)

    manifest = build_manifest(
        output_root=output_root,
        seed=args.seed,
        external_per_class=args.external_per_class,
        canonical_summary=canonical_summary,
        external_summary=external_summary,
        validation=validation,
        outputs=output_frames,
    )
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": "passed",
                "output_root": str(output_root),
                "canonical_rows": len(canonical),
                "split_rows": {
                    "train": len(train),
                    "valid": len(valid),
                    "test": len(test),
                },
                "external_rows": len(external),
                "external_class_counts": class_counts(external),
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
