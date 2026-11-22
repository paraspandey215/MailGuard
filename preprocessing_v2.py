import re
import html

URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE
)

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

SHORTENER_PATTERN = re.compile(
    r"bit\.ly|tinyurl\.com|t\.co|goo\.gl|is\.gd|"
    r"ow\.ly|buff\.ly|rebrand\.ly",
    re.IGNORECASE
)

IP_URL_PATTERN = re.compile(
    r"https?://(?:\d{1,3}\.){3}\d{1,3}",
    re.IGNORECASE
)

def extract_features(email_text):

    raw_text = str(email_text)

    subject_match = re.match(
        r"^\s*Subject\s*:\s*(.*?)(?:\r?\n|$)",
        raw_text,
        re.IGNORECASE
    )

    if subject_match:
        subject = subject_match.group(1)
    else:
        subject = ""

    character_count = len(raw_text)

    words = re.findall(r"\b\w+\b", raw_text)
    word_count = len(words)

    url_count = len(URL_PATTERN.findall(raw_text))
    email_count = len(EMAIL_PATTERN.findall(raw_text))
    shortener_count = len(SHORTENER_PATTERN.findall(raw_text))
    ip_url_count = len(IP_URL_PATTERN.findall(raw_text))

    exclamation_count = raw_text.count("!")
    question_count = raw_text.count("?")

    digit_count = sum(char.isdigit() for char in raw_text)
    uppercase_count = sum(char.isupper() for char in raw_text)
    alphabet_count = sum(char.isalpha() for char in raw_text)

    digit_ratio = (
        digit_count / character_count
        if character_count else 0
    )

    uppercase_ratio = (
        uppercase_count / alphabet_count
        if alphabet_count else 0
    )

    has_html = int(bool(re.search(
        r"<\s*(html|body|div|table|a|script|p)\b",
        raw_text,
        re.IGNORECASE
    )))

    cleaned = re.sub(
        r"<(script|style)\b[^>]*>.*?</\1>",
        " ",
        raw_text,
        flags=re.IGNORECASE | re.DOTALL
    )

    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)

    cleaned = URL_PATTERN.sub(" URLTOKEN ", cleaned)
    cleaned = EMAIL_PATTERN.sub(" EMAILTOKEN ", cleaned)

    cleaned = cleaned.lower()
    cleaned = re.sub(r"[^a-z0-9_\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    cleaned = cleaned[:30000]

    return {
        "clean_text": cleaned,
        "subject_length": len(subject),
        "character_count": character_count,
        "word_count": word_count,
        "url_count": url_count,
        "email_count": email_count,
        "shortener_count": shortener_count,
        "ip_url_count": ip_url_count,
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "digit_ratio": digit_ratio,
        "uppercase_ratio": uppercase_ratio,
        "has_html": has_html
    }

if __name__ == "__main__":

    import os
    import pandas as pd

    print("Loading V2 dataset...")

    df = pd.read_csv(
        "dataset/combined_emails_v2.csv"
    )

    print("Total emails:", len(df))
    print("Extracting advanced features...")

    features = df["text"].apply(extract_features)

    features_df = pd.DataFrame(features.tolist())

    final_df = pd.concat(
        [
            df[["label", "source"]].reset_index(drop=True),
            features_df
        ],
        axis=1
    )

    os.makedirs("dataset/processed", exist_ok=True)

    output_path = (
        "dataset/processed/processed_emails_v2.csv"
    )

    final_df.to_csv(output_path, index=False)

    print("\nADVANCED PREPROCESSING COMPLETED!")
    print("Total records:", len(final_df))
    print("Saved to:", output_path)
