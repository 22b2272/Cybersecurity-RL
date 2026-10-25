"""Extract non-reversible, explainable password composition features."""
import string
import math
from collections import Counter


COMMON_PATTERNS = ("password", "qwerty", "admin", "welcome", "letmein", "123456")


def extract_features(password: str) -> dict[str, int | float]:
    """Describe composition and obvious patterns without retaining the input."""
    password = str(password)
    length = len(password)
    counts = Counter(password)
    entropy = -sum((n / length) * math.log2(n / length) for n in counts.values()) if length else 0.0
    classes = [
        1 if c.islower() else 2 if c.isupper() else 3 if c.isdigit() else 4
        for c in password
    ]
    transitions = sum(left != right for left, right in zip(classes, classes[1:]))
    longest_run = run = 0
    previous = None
    for char in password:
        run = run + 1 if char == previous else 1
        longest_run = max(longest_run, run)
        previous = char
    lowered = password.lower()
    sequential_pairs = sum(
        ord(right) - ord(left) in (-1, 1)
        for left, right in zip(lowered, lowered[1:])
        if left.isalnum() and right.isalnum()
    )
    return {
        "length": len(password),
        "uppercase": sum(c in string.ascii_uppercase for c in password),
        "lowercase": sum(c in string.ascii_lowercase for c in password),
        "digits": sum(c in string.digits for c in password),
        "special": sum(c in string.punctuation for c in password),
        "unique_ratio": len(counts) / length if length else 0.0,
        "shannon_entropy_bits_per_char": entropy,
        "character_class_transitions": transitions,
        "max_repeated_run": longest_run,
        "sequential_adjacent_pairs": sequential_pairs,
        "common_pattern": int(any(pattern in lowered for pattern in COMMON_PATTERNS)),
    }
