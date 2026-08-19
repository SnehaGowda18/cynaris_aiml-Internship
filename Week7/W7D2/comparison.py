from pathlib import Path
import re


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

OUTPUT_DIR = Path("outputs")

BM25_FILE = OUTPUT_DIR / "bm25_results.txt"
DENSE_FILE = OUTPUT_DIR / "dense_results.txt"
COMPARISON_FILE = OUTPUT_DIR / "comparison_results.txt"


# ---------------------------------------------------------
# Extract precision from result file
# ---------------------------------------------------------

def get_precision(file_path):

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return None

    content = file_path.read_text(
        encoding="utf-8"
    )

    match = re.search(
        r"Precision:\s*([\d.]+)%",
        content
    )

    if match:
        return float(match.group(1))

    return None


# ---------------------------------------------------------
# Read BM25 and Dense results
# ---------------------------------------------------------

bm25_precision = get_precision(BM25_FILE)
dense_precision = get_precision(DENSE_FILE)


# ---------------------------------------------------------
# Display comparison
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("BM25 vs DENSE RETRIEVAL COMPARISON")
print("=" * 70)

if bm25_precision is not None:
    print(
        f"BM25 Precision:    "
        f"{bm25_precision:.2f}%"
    )
else:
    print("BM25 Precision: Not available")


if dense_precision is not None:
    print(
        f"Dense Precision:   "
        f"{dense_precision:.2f}%"
    )
else:
    print("Dense Precision: Not available")


# ---------------------------------------------------------
# Determine better method
# ---------------------------------------------------------

if (
    bm25_precision is not None
    and dense_precision is not None
):

    difference = (
        dense_precision - bm25_precision
    )

    print(
        f"\nPrecision Difference: "
        f"{abs(difference):.2f}%"
    )

    if dense_precision > bm25_precision:

        better = "Dense Retrieval"

        print(
            "Better Retrieval Method: "
            "Dense Retrieval"
        )

    elif bm25_precision > dense_precision:

        better = "BM25"

        print(
            "Better Retrieval Method: "
            "BM25"
        )

    else:

        better = "Both"

        print(
            "Better Retrieval Method: "
            "Both performed equally"
        )

else:

    difference = None
    better = "Unavailable"


print("=" * 70)


# ---------------------------------------------------------
# Save comparison evidence
# ---------------------------------------------------------

with open(
    COMPARISON_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "BM25 vs DENSE RETRIEVAL COMPARISON\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )

    if bm25_precision is not None:

        file.write(
            f"BM25 Precision: "
            f"{bm25_precision:.2f}%\n"
        )

    else:

        file.write(
            "BM25 Precision: Not available\n"
        )


    if dense_precision is not None:

        file.write(
            f"Dense Precision: "
            f"{dense_precision:.2f}%\n"
        )

    else:

        file.write(
            "Dense Precision: Not available\n"
        )


    if difference is not None:

        file.write(
            f"Precision Difference: "
            f"{abs(difference):.2f}%\n"
        )

        file.write(
            f"Better Retrieval Method: "
            f"{better}\n"
        )


print(
    f"\nComparison saved to: "
    f"{COMPARISON_FILE}"
)