"""Generate passwords using the operating system cryptographic random source."""
import argparse
import secrets
import string


def generate_password(length: int = 20) -> str:
    """Return a password with at least one character from each character class."""
    if length < 4:
        raise ValueError("Password length must be at least 4")
    groups = [string.ascii_lowercase, string.ascii_uppercase, string.digits, string.punctuation]
    alphabet = "".join(groups)
    chars = [secrets.choice(group) for group in groups]
    chars.extend(secrets.choice(alphabet) for _ in range(length - len(chars)))
    secrets.SystemRandom().shuffle(chars)
    return "".join(chars)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--count", type=int, default=1)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be at least 1")
    for _ in range(args.count):
        print(generate_password(args.length))


if __name__ == "__main__":
    main()
