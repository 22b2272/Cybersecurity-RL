"""Explainable password composition feature extraction."""
import string


def extract_features(password: str) -> dict[str, int]:
    """Count character classes without retaining or logging the password."""
    return {
        "length": len(password),
        "uppercase": sum(c in string.ascii_uppercase for c in password),
        "lowercase": sum(c in string.ascii_lowercase for c in password),
        "digits": sum(c in string.digits for c in password),
        "special": sum(c in string.punctuation for c in password),
    }
