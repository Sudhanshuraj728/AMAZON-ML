from blocking import build_name_index


FILE_PATH = "../dataset/train/train_source2.tsv"


if __name__ == "__main__":

    name_index = build_name_index(
        FILE_PATH,
        chunksize=100_000
    )

    print("\nSample blocks:")

    count = 0

    for key, entity_ids in name_index.items():

        print(
            f"{key} -> "
            f"{len(entity_ids)} records"
        )

        count += 1

        if count == 10:
            break