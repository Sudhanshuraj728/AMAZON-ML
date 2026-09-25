from collections import defaultdict
from data_loader import read_tsv_in_chunks
from normalization import normalize_name, normalize_address, normalize_text


def build_name_index(file_path, chunksize=100_000):
    name_index = defaultdict(list)

    total_rows = 0

    for chunk_number, chunk in enumerate(
        read_tsv_in_chunks(file_path, chunksize),
        start=1
    ):
        chunk["business_name"] = chunk["business_name"].fillna("")
        chunk["country"] = chunk["country"].fillna("")

        for entity_id, name, country in zip(
            chunk["entity_id"],
            chunk["business_name"],
            chunk["country"]
        ):
            name_norm = normalize_name(name)
            country_norm = normalize_text(country)

            if not name_norm:
                continue

            key = (country_norm, name_norm)

            name_index[key].append(entity_id)

        total_rows += len(chunk)

        print(
            f"Name index - chunk {chunk_number}: "
            f"{len(chunk)} rows | "
            f"Total: {total_rows}"
        )

    print("\nName index built.")
    print(f"Total records processed: {total_rows}")
    print(f"Unique name blocks: {len(name_index)}")

    return name_index


def build_address_index(file_path, chunksize=100_000):
    address_index = defaultdict(list)

    total_rows = 0

    for chunk_number, chunk in enumerate(
        read_tsv_in_chunks(file_path, chunksize),
        start=1
    ):
        chunk["business_address"] = chunk["business_address"].fillna("")
        chunk["country"] = chunk["country"].fillna("")

        for entity_id, address, country in zip(
            chunk["entity_id"],
            chunk["business_address"],
            chunk["country"]
        ):
            address_norm = normalize_address(address)
            country_norm = normalize_text(country)

            if not address_norm:
                continue

            key = (country_norm, address_norm)

            address_index[key].append(entity_id)

        total_rows += len(chunk)

        print(
            f"Address index - chunk {chunk_number}: "
            f"{len(chunk)} rows | "
            f"Total: {total_rows}"
        )

    print("\nAddress index built.")
    print(f"Total records processed: {total_rows}")
    print(f"Unique address blocks: {len(address_index)}")

    return address_index


def build_name_token_index(file_path, chunksize=100_000):
    token_index = defaultdict(list)

    total_rows = 0

    for chunk_number, chunk in enumerate(
        read_tsv_in_chunks(file_path, chunksize),
        start=1
    ):
        chunk["business_name"] = chunk["business_name"].fillna("")
        chunk["country"] = chunk["country"].fillna("")

        for entity_id, name, country in zip(
            chunk["entity_id"],
            chunk["business_name"],
            chunk["country"]
        ):
            name_norm = normalize_name(name)
            country_norm = normalize_text(country)

            if not name_norm:
                continue

            tokens = set(name_norm.split())

            for token in tokens:

                # Ignore extremely common/generic short tokens
                if len(token) < 3:
                    continue

                key = (country_norm, token)

                token_index[key].append(entity_id)

        total_rows += len(chunk)

        print(
            f"Token index - chunk {chunk_number}: "
            f"{len(chunk)} rows | "
            f"Total: {total_rows}"
        )

    print("\nName token index built.")
    print(f"Total records processed: {total_rows}")
    print(f"Unique token blocks: {len(token_index)}")

    return token_index