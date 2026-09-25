from blocking import build_address_index


FILE_PATH = "../dataset/train/train_source2.tsv"


if __name__ == "__main__":

    address_index = build_address_index(
        FILE_PATH,
        chunksize=100_000
    )

    print("\nSample address blocks:")

    count = 0

    for key, entity_ids in address_index.items():

        print(f"{key} -> {len(entity_ids)} records")

        count += 1

        if count == 10:
            break