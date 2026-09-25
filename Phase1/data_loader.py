import pandas as pd


DEFAULT_CHUNK_SIZE = 100_000


def read_tsv_in_chunks(file_path, chunksize=DEFAULT_CHUNK_SIZE):
    """
    Read a large TSV file chunk by chunk.

    The complete dataset is processed eventually,
    but only one chunk stays in memory at a time.
    """

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        chunksize=chunksize,
        dtype={
            "entity_id": "string",
            "business_name": "string",
            "business_address": "string",
            "country": "string"
        },
        keep_default_na=True
    ):
        yield chunk