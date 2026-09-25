import os
import csv
import json
import math
from collections import Counter, defaultdict

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")

OUTPUT_DIR = os.path.join(BASE_DIR, "phase0_report")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHUNK_SIZE = 100_000

TRAIN_FILES = {
    "source1": os.path.join(TRAIN_DIR, "train_source1.tsv"),
    "source2": os.path.join(TRAIN_DIR, "train_source2.tsv"),
    "source3": os.path.join(TRAIN_DIR, "train_source3.tsv"),
    "ground_truth": os.path.join(TRAIN_DIR, "train_ground_truth.tsv"),
}

TEST_FILES = {
    "source1": os.path.join(TEST_DIR, "test_source1.tsv"),
    "source2": os.path.join(TEST_DIR, "test_source2.tsv"),
    "source3": os.path.join(TEST_DIR, "test_source3.tsv"),
}


# ============================================================
# HELPERS
# ============================================================

def human_size(size):
    if size < 1024:
        return f"{size} B"

    if size < 1024 ** 2:
        return f"{size / 1024:.2f} KB"

    if size < 1024 ** 3:
        return f"{size / (1024 ** 2):.2f} MB"

    return f"{size / (1024 ** 3):.2f} GB"


def file_size(path):
    if os.path.exists(path):
        return human_size(os.path.getsize(path))
    return "MISSING"


def inspect_schema(path):
    df = pd.read_csv(
        path,
        sep="\t",
        nrows=5,
        dtype=str,
        keep_default_na=False
    )

    return list(df.columns)


def analyze_file(path):
    """
    Memory-safe analysis using chunks.
    """

    print(f"\nAnalyzing: {path}")

    if not os.path.exists(path):
        return {
            "exists": False,
            "error": "File not found"
        }

    total_rows = 0

    missing = Counter()
    country_counts = Counter()

    name_lengths = []
    address_lengths = []

    min_name = math.inf
    max_name = 0
    min_address = math.inf
    max_address = 0

    sample_rows = []

    schema = None

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE
    ):

        if schema is None:
            schema = list(chunk.columns)

        total_rows += len(chunk)

        # ----------------------------------------------------
        # Missing values
        # ----------------------------------------------------

        for col in chunk.columns:

            if col == "entity_id":
                continue

            values = chunk[col]

            missing_count = (
                values.isna() |
                (values.astype(str).str.strip() == "")
            ).sum()

            missing[col] += int(missing_count)

        # ----------------------------------------------------
        # Country
        # ----------------------------------------------------

        if "country" in chunk.columns:
            country_counts.update(
                chunk["country"]
                .astype(str)
                .str.strip()
                .replace("", "[EMPTY]")
            )

        # ----------------------------------------------------
        # Name lengths
        # ----------------------------------------------------

        if "business_name" in chunk.columns:

            lengths = chunk["business_name"].astype(str).str.len()

            if len(lengths) > 0:
                min_name = min(min_name, int(lengths.min()))
                max_name = max(max_name, int(lengths.max()))

                name_lengths.extend(
                    lengths.sample(
                        min(1000, len(lengths)),
                        random_state=42
                    ).tolist()
                )

        # ----------------------------------------------------
        # Address lengths
        # ----------------------------------------------------

        if "business_address" in chunk.columns:

            lengths = chunk["business_address"].astype(str).str.len()

            if len(lengths) > 0:
                min_address = min(min_address, int(lengths.min()))
                max_address = max(max_address, int(lengths.max()))

                address_lengths.extend(
                    lengths.sample(
                        min(1000, len(lengths)),
                        random_state=42
                    ).tolist()
                )

        # ----------------------------------------------------
        # Save first few examples
        # ----------------------------------------------------

        if len(sample_rows) < 10:
            sample_rows.extend(
                chunk.head(10 - len(sample_rows))
                .to_dict("records")
            )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    stats = {
        "exists": True,
        "file_size": file_size(path),
        "rows": total_rows,
        "columns": schema,
        "missing": dict(missing),
        "country_distribution": dict(country_counts),
        "name_length": {
            "min": None if min_name == math.inf else min_name,
            "max": max_name,
            "sample_average": (
                sum(name_lengths) / len(name_lengths)
                if name_lengths else None
            )
        },
        "address_length": {
            "min": None if min_address == math.inf else min_address,
            "max": max_address,
            "sample_average": (
                sum(address_lengths) / len(address_lengths)
                if address_lengths else None
            )
        },
        "sample_rows": sample_rows
    }

    return stats


# ============================================================
# GROUND TRUTH ANALYSIS
# ============================================================

def analyze_ground_truth(path):

    print("\nAnalyzing ground truth...")

    total_s1 = 0

    singleton_count = 0
    non_singleton_count = 0

    s2_match_count = 0
    s3_match_count = 0

    total_matches = 0

    match_count_distribution = Counter()

    both_sources = 0
    only_s2 = 0
    only_s3 = 0

    examples = []

    with open(path, "r", encoding="utf-8") as f:

        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:

            s1 = row["source1_entity_id"]
            raw_matches = row["matched_entity_ids"].strip()

            total_s1 += 1

            if not raw_matches:
                singleton_count += 1
                match_count_distribution[0] += 1
                continue

            ids = [
                x.strip()
                for x in raw_matches.split(",")
                if x.strip()
            ]

            count = len(ids)

            total_matches += count
            match_count_distribution[count] += 1

            has_s2 = any(x.startswith("S2-") for x in ids)
            has_s3 = any(x.startswith("S3-") for x in ids)

            if has_s2:
                s2_match_count += sum(
                    x.startswith("S2-") for x in ids
                )

            if has_s3:
                s3_match_count += sum(
                    x.startswith("S3-") for x in ids
                )

            if has_s2 and has_s3:
                both_sources += 1
            elif has_s2:
                only_s2 += 1
            elif has_s3:
                only_s3 += 1

            non_singleton_count += 1

            if len(examples) < 20:
                examples.append({
                    "source1_entity_id": s1,
                    "matched_entity_ids": ids
                })

    stats = {
        "total_source1_entities": total_s1,

        "singleton_count": singleton_count,

        "singleton_percentage": (
            singleton_count / total_s1 * 100
            if total_s1 else 0
        ),

        "non_singleton_count": non_singleton_count,

        "total_matches": total_matches,

        "average_matches_per_s1": (
            total_matches / total_s1
            if total_s1 else 0
        ),

        "s2_match_count": s2_match_count,
        "s3_match_count": s3_match_count,

        "entities_with_both_s2_and_s3": both_sources,
        "entities_with_only_s2": only_s2,
        "entities_with_only_s3": only_s3,

        "match_count_distribution": dict(
            sorted(match_count_distribution.items())
        ),

        "examples": examples
    }

    return stats


# ============================================================
# COLLECT SMALL SET OF GROUND-TRUTH IDs
# ============================================================

def collect_ground_truth_ids(path, limit=100):

    ids = set()

    with open(path, "r", encoding="utf-8") as f:

        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:

            raw = row["matched_entity_ids"].strip()

            if not raw:
                continue

            for entity_id in raw.split(","):

                entity_id = entity_id.strip()

                if entity_id:
                    ids.add(entity_id)

                if len(ids) >= limit:
                    return ids

    return ids


def find_records_by_ids(path, target_ids):

    found = {}

    if not target_ids:
        return found

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE
    ):

        matches = chunk[
            chunk["entity_id"].isin(target_ids)
        ]

        for row in matches.to_dict("records"):
            found[row["entity_id"]] = row

        if len(found) == len(target_ids):
            break

    return found


# ============================================================
# WRITE REPORT
# ============================================================

def write_report(train_results, test_results, gt_stats):

    report_path = os.path.join(
        OUTPUT_DIR,
        "dataset_summary.txt"
    )

    with open(report_path, "w", encoding="utf-8") as f:

        f.write("=" * 80 + "\n")
        f.write("AMAZON ML CHALLENGE 2026 - PHASE 0 DATASET RECONNAISSANCE\n")
        f.write("=" * 80 + "\n\n")

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        f.write("\nTRAIN DATA\n")
        f.write("-" * 80 + "\n")

        for name, stats in train_results.items():

            f.write(f"\n{name.upper()}\n")

            f.write(f"File size: {stats.get('file_size')}\n")
            f.write(f"Rows: {stats.get('rows')}\n")
            f.write(
                f"Columns: {', '.join(stats.get('columns', []))}\n"
            )

            f.write("\nMissing values:\n")

            for col, count in stats.get("missing", {}).items():

                percentage = (
                    count / stats["rows"] * 100
                    if stats["rows"]
                    else 0
                )

                f.write(
                    f"  {col}: {count:,} "
                    f"({percentage:.4f}%)\n"
                )

            f.write("\nCountry distribution:\n")

            for country, count in sorted(
                stats.get("country_distribution", {}).items(),
                key=lambda x: x[1],
                reverse=True
            ):

                percentage = (
                    count / stats["rows"] * 100
                    if stats["rows"]
                    else 0
                )

                f.write(
                    f"  {country}: {count:,} "
                    f"({percentage:.2f}%)\n"
                )

            f.write("\nName length:\n")
            f.write(
                f"  min = {stats['name_length']['min']}\n"
            )
            f.write(
                f"  max = {stats['name_length']['max']}\n"
            )
            f.write(
                f"  sample avg = "
                f"{stats['name_length']['sample_average']:.2f}\n"
            )

            f.write("\nAddress length:\n")
            f.write(
                f"  min = {stats['address_length']['min']}\n"
            )
            f.write(
                f"  max = {stats['address_length']['max']}\n"
            )
            f.write(
                f"  sample avg = "
                f"{stats['address_length']['sample_average']:.2f}\n"
            )

        # ----------------------------------------------------
        # TEST
        # ----------------------------------------------------

        f.write("\n\nTEST DATA\n")
        f.write("-" * 80 + "\n")

        for name, stats in test_results.items():

            f.write(f"\n{name.upper()}\n")

            f.write(f"File size: {stats.get('file_size')}\n")
            f.write(f"Rows: {stats.get('rows')}\n")
            f.write(
                f"Columns: {', '.join(stats.get('columns', []))}\n"
            )

            f.write("\nMissing values:\n")

            for col, count in stats.get("missing", {}).items():

                percentage = (
                    count / stats["rows"] * 100
                    if stats["rows"]
                    else 0
                )

                f.write(
                    f"  {col}: {count:,} "
                    f"({percentage:.4f}%)\n"
                )

            f.write("\nCountry distribution:\n")

            for country, count in sorted(
                stats.get("country_distribution", {}).items(),
                key=lambda x: x[1],
                reverse=True
            ):

                percentage = (
                    count / stats["rows"] * 100
                    if stats["rows"]
                    else 0
                )

                f.write(
                    f"  {country}: {count:,} "
                    f"({percentage:.2f}%)\n"
                )

        # ----------------------------------------------------
        # GROUND TRUTH
        # ----------------------------------------------------

        f.write("\n\nGROUND TRUTH\n")
        f.write("-" * 80 + "\n")

        f.write(
            f"Total S1 entities: "
            f"{gt_stats['total_source1_entities']:,}\n"
        )

        f.write(
            f"Singletons: "
            f"{gt_stats['singleton_count']:,} "
            f"({gt_stats['singleton_percentage']:.2f}%)\n"
        )

        f.write(
            f"Non-singletons: "
            f"{gt_stats['non_singleton_count']:,}\n"
        )

        f.write(
            f"Total matches: "
            f"{gt_stats['total_matches']:,}\n"
        )

        f.write(
            f"Average matches/S1: "
            f"{gt_stats['average_matches_per_s1']:.4f}\n"
        )

        f.write(
            f"S2 matches: "
            f"{gt_stats['s2_match_count']:,}\n"
        )

        f.write(
            f"S3 matches: "
            f"{gt_stats['s3_match_count']:,}\n"
        )

        f.write(
            f"Entities with both S2 + S3: "
            f"{gt_stats['entities_with_both_s2_and_s3']:,}\n"
        )

        f.write(
            f"Entities with only S2: "
            f"{gt_stats['entities_with_only_s2']:,}\n"
        )

        f.write(
            f"Entities with only S3: "
            f"{gt_stats['entities_with_only_s3']:,}\n"
        )

        f.write("\nMatch-count distribution:\n")

        for count, frequency in gt_stats[
            "match_count_distribution"
        ].items():

            f.write(
                f"  {count} matches: "
                f"{frequency:,} S1 entities\n"
            )

        # ----------------------------------------------------
        # GROUND TRUTH EXAMPLES
        # ----------------------------------------------------

        f.write("\n\nGROUND TRUTH EXAMPLES\n")
        f.write("-" * 80 + "\n")

        for example in gt_stats["examples"]:

            f.write(
                f"{example['source1_entity_id']}\t"
                f"{','.join(example['matched_entity_ids'])}\n"
            )

    print(f"\nReport written to:\n{report_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026")
    print("PHASE 0 - DATASET RECONNAISSANCE")
    print("=" * 80)

    print("\nBase directory:")
    print(BASE_DIR)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    print("\nChecking files...")

    all_files = {
        **TRAIN_FILES,
        **TEST_FILES
    }

    for name, path in all_files.items():

        status = "OK" if os.path.exists(path) else "MISSING"

        print(
            f"{name:15} "
            f"{status:8} "
            f"{file_size(path)}"
        )

    # --------------------------------------------------------
    # Train analysis
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("TRAIN DATA ANALYSIS")
    print("=" * 80)

    train_results = {}

    for name in ["source1", "source2", "source3"]:

        train_results[name] = analyze_file(
            TRAIN_FILES[name]
        )

    # --------------------------------------------------------
    # Test analysis
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("TEST DATA ANALYSIS")
    print("=" * 80)

    test_results = {}

    for name in ["source1", "source2", "source3"]:

        test_results[name] = analyze_file(
            TEST_FILES[name]
        )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("GROUND TRUTH ANALYSIS")
    print("=" * 80)

    gt_stats = analyze_ground_truth(
        TRAIN_FILES["ground_truth"]
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    full_results = {
        "train": train_results,
        "test": test_results,
        "ground_truth": gt_stats
    }

    json_path = os.path.join(
        OUTPUT_DIR,
        "phase0_results.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            full_results,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Write readable report
    # --------------------------------------------------------

    write_report(
        train_results,
        test_results,
        gt_stats
    )

    # --------------------------------------------------------
    # Ground-truth sample IDs
    # --------------------------------------------------------

    print("\nCollecting sample ground-truth IDs...")

    sample_ids = collect_ground_truth_ids(
        TRAIN_FILES["ground_truth"],
        limit=100
    )

    print(
        f"Collected {len(sample_ids)} sample matched IDs."
    )

    # Find corresponding records
    s2_records = find_records_by_ids(
        TRAIN_FILES["source2"],
        {
            x for x in sample_ids
            if x.startswith("S2-")
        }
    )

    s3_records = find_records_by_ids(
        TRAIN_FILES["source3"],
        {
            x for x in sample_ids
            if x.startswith("S3-")
        }
    )

    samples_path = os.path.join(
        OUTPUT_DIR,
        "matched_record_samples.txt"
    )

    with open(
        samples_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "SAMPLE RECORDS REFERENCED BY GROUND TRUTH\n"
        )

        f.write("=" * 80 + "\n\n")

        f.write("SOURCE 2\n")
        f.write("-" * 80 + "\n")

        for entity_id, record in s2_records.items():

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
            )

            f.write("\n")

        f.write("\nSOURCE 3\n")
        f.write("-" * 80 + "\n")

        for entity_id, record in s3_records.items():

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
            )

            f.write("\n")

    print(
        f"Sample records written to:\n{samples_path}"
    )

    print("\n" + "=" * 80)
    print("PHASE 0 COMPLETE")
    print("=" * 80)

    print("\nGenerated files:")

    for filename in os.listdir(OUTPUT_DIR):
        print(
            "  ",
            os.path.join(OUTPUT_DIR, filename)
        )


if __name__ == "__main__":
    main()