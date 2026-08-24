from pathlib import Path
import re


BASIC_FILE = Path("outputs/basic_results.txt")
CHROMA_FILE = Path("outputs/chroma_results.txt")


def extract_latencies(file_path):
    """Extract latency values from a result file."""
    if not file_path.exists():
        return []

    text = file_path.read_text(encoding="utf-8")

    values = re.findall(
        r"Latency:\s*([0-9.]+)\s*seconds",
        text
    )

    return [float(value) for value in values]


def calculate_average(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


print("=" * 70)
print("W7D3 - LlamaIndex vs ChromaDB Comparison")
print("=" * 70)


basic_latencies = extract_latencies(BASIC_FILE)
chroma_latencies = extract_latencies(CHROMA_FILE)


print("\nLlamaIndex Basic")
print("-" * 70)

print(f"Queries completed: {len(basic_latencies)}")

if basic_latencies:
    print(
        f"Average latency: "
        f"{calculate_average(basic_latencies):.4f} seconds"
    )

    print(
        f"Fastest query: "
        f"{min(basic_latencies):.4f} seconds"
    )

    print(
        f"Slowest query: "
        f"{max(basic_latencies):.4f} seconds"
    )


print("\nLlamaIndex + ChromaDB")
print("-" * 70)

print(f"Queries completed: {len(chroma_latencies)}")

if chroma_latencies:
    print(
        f"Average latency: "
        f"{calculate_average(chroma_latencies):.4f} seconds"
    )

    print(
        f"Fastest query: "
        f"{min(chroma_latencies):.4f} seconds"
    )

    print(
        f"Slowest query: "
        f"{max(chroma_latencies):.4f} seconds"
    )


if basic_latencies and chroma_latencies:

    basic_average = calculate_average(
        basic_latencies
    )

    chroma_average = calculate_average(
        chroma_latencies
    )

    difference = basic_average - chroma_average

    print("\nComparison")
    print("-" * 70)

    if difference > 0:
        print(
            f"ChromaDB was faster by "
            f"{difference:.4f} seconds on average."
        )

    elif difference < 0:
        print(
            f"Basic LlamaIndex was faster by "
            f"{abs(difference):.4f} seconds on average."
        )

    else:
        print(
            "Both approaches had the same "
            "average latency."
        )

else:

    print(
        "\nComparison could not be calculated. "
        "Check the result files."
    )