from blocking import (
    build_source_name_index,
    build_source_address_index
)


S2_PATH = "../dataset/train/train_source2.tsv"
S3_PATH = "../dataset/train/train_source3.tsv"


if __name__ == "__main__":

    print("=" * 60)
    print("BUILDING S2 NAME INDEX")
    print("=" * 60)

    s2_name_index = build_source_name_index(
        S2_PATH,
        "S2",
        chunksize=100_000
    )

    print("\n")
    print("=" * 60)
    print("BUILDING S3 NAME INDEX")
    print("=" * 60)

    s3_name_index = build_source_name_index(
        S3_PATH,
        "S3",
        chunksize=100_000
    )

    print("\n")
    print("=" * 60)
    print("TESTING COMMON NAME BLOCK")
    print("=" * 60)

    # Find a block that exists in S2
    # and inspect its contents.

    count = 0

    for key, entity_ids in s2_name_index.items():

        print(f"\nBlock: {key}")
        print(f"Number of records: {len(entity_ids)}")
        print(f"Sample: {entity_ids[:5]}")

        count += 1

        if count == 5:
            break

    print("\nTest completed.")