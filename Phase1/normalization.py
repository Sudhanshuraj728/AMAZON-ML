import re
import unicodedata


# ---------------------------------------------------------
# Basic text normalization
# ---------------------------------------------------------

def normalize_text(text):
    """
    General normalization for business names and addresses.

    Keeps the meaning of the original text as much as possible.
    """

    if text is None:
        return ""

    text = str(text)

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Case normalization
    text = text.casefold()

    # Replace & with "and"
    text = text.replace("&", " and ")

    # Replace punctuation/special characters with spaces
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Normalize underscores as spaces
    text = text.replace("_", " ")

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------
# Business name normalization
# ---------------------------------------------------------

def normalize_name(name):
    """
    Normalize business name.
    """

    return normalize_text(name)


# ---------------------------------------------------------
# Address normalization
# ---------------------------------------------------------

def normalize_address(address):
    """
    Normalize business address.
    """

    if address is None:
        return ""

    return normalize_text(address)


# ---------------------------------------------------------
# Tokenization
# ---------------------------------------------------------

def tokenize(text):
    """
    Convert normalized text into tokens.
    """

    if not text:
        return []

    return text.split()


# ---------------------------------------------------------
# Digit extraction
# ---------------------------------------------------------

def extract_digits(text):
    """
    Extract all numeric sequences from text.

    Example:
        '323 80th Street' -> ['323', '80']
    """

    if not text:
        return []

    return re.findall(r"\d+", text)


# ---------------------------------------------------------
# Character representation
# ---------------------------------------------------------

def alphanumeric_only(text):
    """
    Keep only letters and numbers.

    Example:
        'ABC Technologies Pvt. Ltd.'
        ->
        'abctechnologiespvtltd'
    """

    if not text:
        return ""

    return re.sub(r"[^\w]", "", text, flags=re.UNICODE)


# ---------------------------------------------------------
# Complete record normalization
# ---------------------------------------------------------

def normalize_record(record):
    """
    Normalize one business record.

    Input:
        dict-like record containing:
        business_name
        business_address
        country

    Output:
        dictionary containing derived fields.
    """

    raw_name = record.get("business_name", "")
    raw_address = record.get("business_address", "")
    country = record.get("country", "")

    name_norm = normalize_name(raw_name)
    address_norm = normalize_address(raw_address)

    return {
        "name_norm": name_norm,
        "address_norm": address_norm,

        "name_tokens": tokenize(name_norm),
        "address_tokens": tokenize(address_norm),

        "name_digits": extract_digits(name_norm),
        "address_digits": extract_digits(address_norm),

        "name_alnum": alphanumeric_only(name_norm),
        "address_alnum": alphanumeric_only(address_norm),

        "country_norm": normalize_text(country),
    }


# ---------------------------------------------------------
# Small manual test
# ---------------------------------------------------------

if __name__ == "__main__":

    test_record = {
        "business_name": "ABC Technologies Pvt. Ltd.",
        "business_address": "323, 80TH ST., Bangalore",
        "country": "India"
    }

    result = normalize_record(test_record)

    print("\nOriginal:")
    print(test_record)

    print("\nNormalized:")
    for key, value in result.items():
        print(f"{key}: {value}")