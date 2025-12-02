#!/usr/bin/env python
"""Parse HMDB XML and export to JSON format for evaluation.

Supports reading from zip files and uses streaming for memory efficiency.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from tqdm import tqdm


def get_text(element: ET.Element, tag_suffix: str) -> str | None:
    """Safely extract text from XML element, handling namespaces."""
    for child in element:
        if child.tag.endswith(tag_suffix):
            return child.text
    return None


def parse_hmdb_xml(
    input_path: str,
    output_path: str,
    limit: int | None = None,
) -> None:
    """Parse HMDB XML and export to JSON.

    Supports both raw XML files and zip archives containing XML.
    Uses streaming parser for memory efficiency with large files.

    Args:
        input_path: Path to HMDB XML file or zip archive.
        output_path: Output path for JSON file.
        limit: Maximum number of metabolites to process.
    """
    input_path = Path(input_path)

    # Open file handle (supports both zip and raw XML)
    if input_path.suffix == ".zip":
        print(f"Opening zip archive: {input_path}")
        zf = zipfile.ZipFile(input_path, "r")
        # Find the XML file in the archive
        xml_files = [n for n in zf.namelist() if n.endswith(".xml")]
        if not xml_files:
            raise ValueError(f"No XML files found in {input_path}")
        xml_name = xml_files[0]
        print(f"Reading {xml_name} from archive...")
        file_handle = zf.open(xml_name)
    else:
        print(f"Opening XML file: {input_path}")
        file_handle = open(input_path, "rb")
        zf = None

    metabolites: list[dict] = []
    processed = 0

    try:
        # Use iterparse for memory-efficient streaming
        context = ET.iterparse(file_handle, events=("end",))

        pbar = tqdm(desc="Parsing metabolites", unit=" compounds")

        for event, elem in context:
            # Handle namespace - check if tag ends with 'metabolite'
            if elem.tag.endswith("metabolite"):
                # Extract fields
                hmdb_id = get_text(elem, "accession")
                name = get_text(elem, "name")

                if hmdb_id and name:
                    # Get synonyms
                    synonyms: list[str] = []
                    for child in elem:
                        if child.tag.endswith("synonyms"):
                            for syn in child:
                                if syn.tag.endswith("synonym") and syn.text:
                                    synonyms.append(syn.text.strip())
                        # Also get IUPAC names as synonyms
                        elif child.tag.endswith("iupac_name") and child.text:
                            iupac = child.text.strip()
                            if iupac and iupac not in synonyms:
                                synonyms.append(iupac)
                        elif child.tag.endswith("traditional_iupac") and child.text:
                            trad = child.text.strip()
                            if trad and trad not in synonyms:
                                synonyms.append(trad)

                    metabolites.append({
                        "hmdb_id": hmdb_id,
                        "name": name,
                        "synonyms": synonyms,
                    })
                    processed += 1
                    pbar.update(1)

                    if limit and processed >= limit:
                        break

                # Clear element to free memory
                elem.clear()

        pbar.close()

    finally:
        file_handle.close()
        if zf:
            zf.close()

    # Save to JSON
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(metabolites)} metabolites to {output_path}...")

    with open(output_path, "w") as f:
        json.dump(metabolites, f, indent=2)

    print(f"Done! Exported {len(metabolites)} metabolites")

    # Print some stats
    total_synonyms = sum(len(m["synonyms"]) for m in metabolites)
    avg_synonyms = total_synonyms / len(metabolites) if metabolites else 0
    print(f"  Total synonyms: {total_synonyms}")
    print(f"  Average synonyms per metabolite: {avg_synonyms:.1f}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Parse HMDB XML and export to JSON"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="HMDB XML file or zip archive path"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of metabolites to process"
    )
    args = parser.parse_args()

    parse_hmdb_xml(args.input, args.output, args.limit)


if __name__ == "__main__":
    main()
