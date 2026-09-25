from collections import Counter
from data_loader import read_tsv_in_chunks
from normalization import normalize_name


FILE_PATH = "../dataset/train/train_source2.tsv"


def analyze_name_tokens(file_path, chunksize=100_000):

    token_frequency = Counter()

    total_rows = 0

    for chunk_number, chunk in enumerate(
        read_tsv_in_chunks(file_path, chunksize),
        start=1
    ):

        chunk["business_name"] = chunk["business_name"].fillna("")

        for name in chunk["business_name"]:

            name_norm = normalize_name(name)

            if not name_norm:
                continue

            # Set so repeated token in same name is counted once
            tokens = set(name_norm.split())

            for token in tokens:

                if len(token) >= 3:
                    token_frequency[token] += 1

        total_rows += len(chunk)

        print(
            f"Token analysis - chunk {chunk_number}: "
            f"{len(chunk)} rows | "
            f"Total: {total_rows}"
        )

    print("\nToken analysis completed.")

    print(f"Total records: {total_rows}")
    print(f"Unique tokens: {len(token_frequency)}")

    print("\nMost common tokens:")

    for token, count in token_frequency.most_common(30):
        print(f"{token:30} -> {count}")

    return token_frequency


if __name__ == "__main__":
    analyze_name_tokens(
        FILE_PATH,
        chunksize=100_000
    )