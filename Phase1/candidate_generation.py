from normalization import normalize_name, normalize_text


def generate_name_candidates(
    s1_record,
    s2_name_index,
    s3_name_index
):
    """
    Generate candidates using exact normalized business name
    + country.

    Returns:
        set of tuples:
        {
            ("S2", "S2-123"),
            ("S3", "S3-456")
        }
    """

    candidates = set()

    # -----------------------------
    # Normalize S1 record
    # -----------------------------

    name = s1_record.get("business_name", "")
    country = s1_record.get("country", "")

    name_norm = normalize_name(name)
    country_norm = normalize_text(country)

    # No usable name
    if not name_norm:
        return candidates

    key = (country_norm, name_norm)

    # -----------------------------
    # Search S2
    # -----------------------------

    if key in s2_name_index:
        for candidate in s2_name_index[key]:
            candidates.add(candidate)

    # -----------------------------
    # Search S3
    # -----------------------------

    if key in s3_name_index:
        for candidate in s3_name_index[key]:
            candidates.add(candidate)

    return candidates