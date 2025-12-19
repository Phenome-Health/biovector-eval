"""Add Arivale real-world metabolite names to ground truth dataset."""

import csv
import json
from pathlib import Path


def add_arivale_ground_truth():
    """Add arivale entries to ground truth JSON and TSV files."""
    data_dir = Path("data/hmdb")

    # Load existing metabolites to validate HMDB IDs
    with open(data_dir / "metabolites.json") as f:
        metabolites = json.load(f)
    hmdb_ids = {m["hmdb_id"] for m in metabolites}

    # Load existing ground truth
    with open(data_dir / "ground_truth.json") as f:
        gt = json.load(f)

    # Load arivale data
    with open(data_dir / "arivale_ground_truth.tsv") as f:
        reader = csv.DictReader(f, delimiter="\t")
        arivale = list(reader)

    # Add arivale entries
    added = 0
    skipped = 0
    for row in arivale:
        # Convert HMDB format: HMDB00001 -> HMDB0000001
        hmdb_id = "HMDB" + row["HMDB"][4:].zfill(7)

        # Skip if not in metabolites database
        if hmdb_id not in hmdb_ids:
            skipped += 1
            continue

        gt["queries"].append({
            "query": row["BIOCHEMICAL_NAME"],
            "expected": [hmdb_id],
            "category": "arivale",
            "difficulty": "medium",
            "notes": "Real-world name from Arivale dataset",
        })
        added += 1

    # Update metadata
    gt["metadata"]["total_queries"] = len(gt["queries"])
    gt["metadata"]["categories"] = list({q["category"] for q in gt["queries"]})

    # Save JSON
    with open(data_dir / "ground_truth.json", "w") as f:
        json.dump(gt, f, indent=2)

    # Save TSV
    with open(data_dir / "ground_truth.tsv", "w") as f:
        f.write("query\texpected\tcategory\tdifficulty\tnotes\n")
        for q in gt["queries"]:
            expected = ",".join(q["expected"])
            notes = q.get("notes", "").replace("\t", " ")
            f.write(f"{q['query']}\t{expected}\t{q['category']}\t{q['difficulty']}\t{notes}\n")

    print(f"Added {added} arivale entries to ground truth")
    print(f"Skipped {skipped} entries (HMDB ID not in database)")
    print(f"Total queries: {len(gt['queries'])}")


if __name__ == "__main__":
    add_arivale_ground_truth()
