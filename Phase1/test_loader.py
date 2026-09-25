from data_loader import read_tsv_in_chunks


FILE_PATH = "../dataset/train/train_source1.tsv"


if __name__ == "__main__":

    total_rows = 0
    chunk_number = 0

    for chunk in read_tsv_in_chunks(FILE_PATH, chunksize=100_000):

        chunk_number += 1
        total_rows += len(chunk)

        print(
            f"Chunk {chunk_number}: "
            f"{len(chunk)} rows | "
            f"Total processed: {total_rows}"
        )

        # Only testing first 3 chunks for now
        if chunk_number == 3:
            break

    print("\nTest completed.")