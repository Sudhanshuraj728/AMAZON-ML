from blocking import build_informative_token_index


FILE_PATH = "../dataset/train/train_source2.tsv"


if __name__ == "__main__":

    token_index = build_informative_token_index(
        FILE_PATH,
        max_frequency=10000,
        chunksize=100_000
    )

    print("\nSample informative token blocks:")

    count = 0

    for key, entity_ids in token_index.items():

        print(
            f"{key} -> "
            f"{len(entity_ids)} records"
        )

        count += 1

        if count == 20:
            break