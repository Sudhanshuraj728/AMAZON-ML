from blocking import (
    build_source_name_index
)

from candidate_generation import generate_name_candidates


S2_PATH = "../dataset/train/train_source2.tsv"
S3_PATH = "../dataset/train/train_source3.tsv"


if __name__ == "__main__":

    print("Building S2 name index...")

    s2_name_index = build_source_name_index(
        S2_PATH,
        "S2",
        chunksize=100_000
    )

    print("\nBuilding S3 name index...")

    s3_name_index = build_source_name_index(
        S3_PATH,
        "S3",
        chunksize=100_000
    )

    # ---------------------------------
    # Test S1 record
    # ---------------------------------

    test_s1_record = {
        "business_name": "Summit Inc",
        "business_address": "",
        "country": "US"
    }

    candidates = generate_name_candidates(
        test_s1_record,
        s2_name_index,
        s3_name_index
    )

    print("\n" + "=" * 60)
    print("CANDIDATE GENERATION TEST")
    print("=" * 60)

    print(f"S1 name: {test_s1_record['business_name']}")
    print(f"S1 country: {test_s1_record['country']}")

    print(f"\nTotal candidates: {len(candidates)}")

    # Count by source
    s2_candidates = [
        candidate
        for candidate in candidates
        if candidate[0] == "S2"
    ]

    s3_candidates = [
        candidate
        for candidate in candidates
        if candidate[0] == "S3"
    ]

    print(f"S2 candidates: {len(s2_candidates)}")
    print(f"S3 candidates: {len(s3_candidates)}")

    print("\nSample candidates:")

    for candidate in list(candidates)[:20]:
        print(candidate)

    print("\nTest completed.")