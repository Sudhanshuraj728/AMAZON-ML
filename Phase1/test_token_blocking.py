from blocking import build_name_token_index


FILE_PATH = "../dataset/train/train_source2.tsv"


if __name__ == "__main__":

    token_index = build_name_token_index(
        FILE_PATH,
        chunksize=100_000
    )

    print("\nSample token blocks:")

    count = 0

    for key, entity_ids in token_index.items():

        print(f"{key} -> {len(entity_ids)} records")

        count += 1

        if count == 20:
            break