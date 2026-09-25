import os
import csv
import json
from collections import defaultdict, Counter

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
OUTPUT_DIR = os.path.join(BASE_DIR, "phase0_report")

os.makedirs(OUTPUT_DIR, exist_ok=True)

GROUND_TRUTH = os.path.join(
    TRAIN_DIR,
    "train_ground_truth.tsv"
)

SOURCE1 = os.path.join(
    TRAIN_DIR,
    "train_source1.tsv"
)

SOURCE2 = os.path.join(
    TRAIN_DIR,
    "train_source2.tsv"
)

SOURCE3 = os.path.join(
    TRAIN_DIR,
    "train_source3.tsv"
)

CHUNK_SIZE = 100_000

# Number of S1 entities sampled from each match-count bucket
PER_BUCKET = 500

# Maximum S1 entities in final sample
MAX_S1 = 4000


# ============================================================
# STEP 1
# READ GROUND TRUTH AND CREATE STRATIFIED SAMPLE
# ============================================================

def create_sample():

    print("=" * 80)
    print("STEP 1 - SAMPLING GROUND TRUTH")
    print("=" * 80)

    buckets = defaultdict(list)

    with open(
        GROUND_TRUTH,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        for row in reader:

            s1_id = row["source1_entity_id"]

            raw = row["matched_entity_ids"].strip()

            if raw:
                matches = [
                    x.strip()
                    for x in raw.split(",")
                    if x.strip()
                ]
            else:
                matches = []

            match_count = len(matches)

            # Store only up to PER_BUCKET
            # for each cardinality bucket.
            if len(buckets[match_count]) < PER_BUCKET:
                buckets[match_count].append(
                    {
                        "source1_entity_id": s1_id,
                        "matched_entity_ids": matches
                    }
                )

    # --------------------------------------------------------
    # Build final sample
    # --------------------------------------------------------

    sample = []

    for match_count in sorted(buckets):

        rows = buckets[match_count]

        print(
            f"{match_count:2d} matches -> "
            f"{len(rows):4d} sampled"
        )

        sample.extend(rows)

    # Safety limit
    sample = sample[:MAX_S1]

    print("\nTotal sampled S1:", len(sample))

    return sample


# ============================================================
# STEP 2
# COLLECT REQUIRED IDS
# ============================================================

def collect_required_ids(sample):

    s1_ids = set()
    s2_ids = set()
    s3_ids = set()

    for row in sample:

        s1_ids.add(
            row["source1_entity_id"]
        )

        for entity_id in row["matched_entity_ids"]:

            if entity_id.startswith("S2-"):
                s2_ids.add(entity_id)

            elif entity_id.startswith("S3-"):
                s3_ids.add(entity_id)

    print("\nRequired records:")
    print("S1:", len(s1_ids))
    print("S2:", len(s2_ids))
    print("S3:", len(s3_ids))

    return s1_ids, s2_ids, s3_ids


# ============================================================
# STEP 3
# FIND RECORDS IN LARGE TSV
# ============================================================

def find_records(
    path,
    target_ids,
    source_name
):

    print("\nScanning:", source_name)

    found = {}

    if not target_ids:
        return found

    chunk_number = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE
    ):

        chunk_number += 1

        matches = chunk[
            chunk["entity_id"].isin(target_ids)
        ]

        if not matches.empty:

            for record in matches.to_dict(
                orient="records"
            ):

                found[
                    record["entity_id"]
                ] = record

        if chunk_number % 10 == 0:

            print(
                f"  processed "
                f"{chunk_number:,} chunks | "
                f"found {len(found):,}/{len(target_ids):,}"
            )

        if len(found) == len(target_ids):
            break

    print(
        f"Completed {source_name}: "
        f"{len(found):,}/{len(target_ids):,}"
    )

    return found


# ============================================================
# STEP 4
# BUILD PAIRWISE FORENSIC DATASET
# ============================================================

def build_forensic_rows(
    sample,
    s1_records,
    s2_records,
    s3_records
):

    rows = []

    for item in sample:

        s1_id = item["source1_entity_id"]

        s1 = s1_records.get(s1_id)

        if s1 is None:
            continue

        matches = item["matched_entity_ids"]

        # Singleton
        if not matches:

            rows.append(
                {
                    "source1_entity_id": s1_id,
                    "match_count": 0,

                    "s1_name": s1["business_name"],
                    "s1_address": s1["business_address"],
                    "s1_country": s1["country"],

                    "matched_entity_id": "",
                    "matched_source": "",

                    "matched_name": "",
                    "matched_address": "",
                    "matched_country": ""
                }
            )

            continue

        # Non-singleton
        for entity_id in matches:

            if entity_id.startswith("S2-"):
                record = s2_records.get(entity_id)
                source = "S2"

            else:
                record = s3_records.get(entity_id)
                source = "S3"

            if record is None:
                continue

            rows.append(
                {
                    "source1_entity_id": s1_id,
                    "match_count": len(matches),

                    "s1_name": s1["business_name"],
                    "s1_address": s1["business_address"],
                    "s1_country": s1["country"],

                    "matched_entity_id": entity_id,
                    "matched_source": source,

                    "matched_name": record["business_name"],
                    "matched_address": record["business_address"],
                    "matched_country": record["country"]
                }
            )

    return rows


# ============================================================
# STEP 5
# BASIC STATISTICS
# ============================================================

def calculate_statistics(rows):

    print("\n" + "=" * 80)
    print("STEP 5 - FORENSIC STATISTICS")
    print("=" * 80)

    total_pairs = len(rows)

    source_counts = Counter()
    country_counts = Counter()

    empty_s1_addresses = 0
    empty_match_addresses = 0

    exact_name = 0
    exact_address = 0

    for row in rows:

        source_counts[
            row["matched_source"]
        ] += 1

        country_counts[
            row["s1_country"]
        ] += 1

        s1_name = row["s1_name"].strip().lower()
        m_name = row["matched_name"].strip().lower()

        s1_address = row["s1_address"].strip().lower()
        m_address = row["matched_address"].strip().lower()

        if not s1_address:
            empty_s1_addresses += 1

        if row["matched_entity_id"] and not m_address:
            empty_match_addresses += 1

        if (
            row["matched_entity_id"]
            and s1_name == m_name
        ):
            exact_name += 1

        if (
            row["matched_entity_id"]
            and s1_address
            and m_address
            and s1_address == m_address
        ):
            exact_address += 1

    matched_pairs = [
        r for r in rows
        if r["matched_entity_id"]
    ]

    n = len(matched_pairs)

    stats = {
        "total_forensic_rows": total_pairs,

        "matched_pairs": n,

        "source_distribution": dict(
            source_counts
        ),

        "country_distribution": dict(
            country_counts
        ),

        "exact_raw_name_rate": (
            exact_name / n
            if n else 0
        ),

        "exact_raw_address_rate": (
            exact_address / n
            if n else 0
        ),

        "empty_s1_address_rate": (
            empty_s1_addresses / total_pairs
            if total_pairs else 0
        ),

        "empty_matched_address_rate": (
            empty_match_addresses / n
            if n else 0
        )
    }

    return stats


# ============================================================
# STEP 6
# WRITE JSONL
# ============================================================

def write_jsonl(rows):

    path = os.path.join(
        OUTPUT_DIR,
        "forensic_pairs.jsonl"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
            )

            f.write("\n")

    print(
        "\nForensic pairs written to:"
    )

    print(path)

    return path


# ============================================================
# STEP 7
# WRITE HUMAN-READABLE REPORT
# ============================================================

def write_report(
    sample,
    rows,
    stats
):

    path = os.path.join(
        OUTPUT_DIR,
        "forensic_report.txt"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "=" * 80 + "\n"
        )

        f.write(
            "AMAZON ML 2026 - "
            "GROUND TRUTH FORENSICS\n"
        )

        f.write(
            "=" * 80 + "\n\n"
        )

        f.write(
            f"Sampled S1 entities: "
            f"{len(sample):,}\n"
        )

        f.write(
            f"Forensic matched rows: "
            f"{stats['matched_pairs']:,}\n\n"
        )

        f.write(
            "SOURCE DISTRIBUTION\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

        for source, count in sorted(
            stats["source_distribution"].items()
        ):

            f.write(
                f"{source}: {count:,}\n"
            )

        f.write(
            "\nCOUNTRY DISTRIBUTION\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

        for country, count in sorted(
            stats["country_distribution"].items()
        ):

            f.write(
                f"{country}: {count:,}\n"
            )

        f.write(
            "\nRAW EXACT MATCH RATES\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

        f.write(
            f"Exact raw name: "
            f"{stats['exact_raw_name_rate']:.4%}\n"
        )

        f.write(
            f"Exact raw address: "
            f"{stats['exact_raw_address_rate']:.4%}\n"
        )

        f.write(
            f"Empty S1 address: "
            f"{stats['empty_s1_address_rate']:.4%}\n"
        )

        f.write(
            f"Empty matched address: "
            f"{stats['empty_matched_address_rate']:.4%}\n"
        )

        # ----------------------------------------------------
        # Actual examples
        # ----------------------------------------------------

        f.write(
            "\n\nACTUAL MATCH EXAMPLES\n"
        )

        f.write(
            "=" * 80 + "\n"
        )

        # Take first 100 matched pairs
        count = 0

        current_s1 = None

        for row in rows:

            if not row["matched_entity_id"]:
                continue

            if current_s1 != row[
                "source1_entity_id"
            ]:

                current_s1 = row[
                    "source1_entity_id"
                ]

                f.write("\n")
                f.write(
                    f"S1: {current_s1}\n"
                )

                f.write(
                    f"Name: {row['s1_name']}\n"
                )

                f.write(
                    f"Address: "
                    f"{row['s1_address']}\n"
                )

                f.write(
                    f"Country: "
                    f"{row['s1_country']}\n"
                )

                f.write(
                    "-" * 60 + "\n"
                )

            f.write(
                f"{row['matched_source']} "
                f"{row['matched_entity_id']}\n"
            )

            f.write(
                f"  Name: "
                f"{row['matched_name']}\n"
            )

            f.write(
                f"  Address: "
                f"{row['matched_address']}\n"
            )

            f.write(
                f"  Country: "
                f"{row['matched_country']}\n"
            )

            count += 1

            if count >= 150:
                break

    print(
        "\nForensic report written to:"
    )

    print(path)

    return path


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026")
    print("PHASE 0.2 - GROUND TRUTH FORENSICS")
    print("=" * 80)

    # --------------------------------------------------------
    # Sample ground truth
    # --------------------------------------------------------

    sample = create_sample()

    # --------------------------------------------------------
    # Required IDs
    # --------------------------------------------------------

    s1_ids, s2_ids, s3_ids = (
        collect_required_ids(sample)
    )

    # --------------------------------------------------------
    # Retrieve records
    # --------------------------------------------------------

    s1_records = find_records(
        SOURCE1,
        s1_ids,
        "SOURCE 1"
    )

    s2_records = find_records(
        SOURCE2,
        s2_ids,
        "SOURCE 2"
    )

    s3_records = find_records(
        SOURCE3,
        s3_ids,
        "SOURCE 3"
    )

    # --------------------------------------------------------
    # Build pair dataset
    # --------------------------------------------------------

    rows = build_forensic_rows(
        sample,
        s1_records,
        s2_records,
        s3_records
    )

    print(
        "\nTotal forensic rows:",
        len(rows)
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    stats = calculate_statistics(rows)

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    stats_path = os.path.join(
        OUTPUT_DIR,
        "forensic_statistics.json"
    )

    with open(
        stats_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            stats,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Save pair dataset
    # --------------------------------------------------------

    write_jsonl(rows)

    # --------------------------------------------------------
    # Human report
    # --------------------------------------------------------

    write_report(
        sample,
        rows,
        stats
    )

    print("\n" + "=" * 80)
    print("PHASE 0.2 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()