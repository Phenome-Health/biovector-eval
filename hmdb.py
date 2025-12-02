from pathlib import Path
from typing import List, Optional, AsyncGenerator, cast, Any
import xml.etree.ElementTree as ET
import logging
from tqdm.asyncio import tqdm

from ..schemas.domain_schema import DomainDocument
from .base import BaseDataProcessor

logger = logging.getLogger(__name__)


class HMDBProcessor(BaseDataProcessor):
    """Process HMDB metabolite XML data, focusing on names and identifiers."""

    xml_file: Path
    _total_compounds: int

    def __init__(self, xml_file: Path, skip_count: bool = True) -> None:
        """Initialize with path to HMDB XML file.

        Args:
            xml_file: Path to HMDB metabolites XML file
            skip_count: Skip counting compounds for faster initialization
        """
        self.xml_file = xml_file
        if skip_count:
            self._total_compounds = 0  # Unknown, but avoid slow counting
            logger.info(
                f"Skipping compound count for {xml_file} (faster initialization)"
            )
        else:
            self._total_compounds = self._count_compounds()
            logger.info(f"Found {self._total_compounds} compounds in {xml_file}")

    def _count_compounds(self) -> int:
        """Count total number of compounds in XML file using iterative parsing."""
        count = 0
        try:
            # Use iterparse for memory-efficient counting
            for event, elem in ET.iterparse(self.xml_file, events=("end",)):
                # Handle namespace - check if tag ends with 'metabolite'
                if elem.tag.endswith("metabolite"):
                    count += 1
                    # Clear the element to save memory
                    elem.clear()
            return count
        except Exception as e:
            logger.error(f"Error counting compounds: {e}")
            return 0

    @staticmethod
    def _get_text(
        element: ET.Element, tag_name: str, default: Optional[str] = None
    ) -> str:
        """Safely extract text from XML element, handling namespaces."""
        try:
            # Look for any element with tag ending in the given name (handles namespace)
            for child in element:
                if child.tag.endswith(tag_name):
                    return child.text if child.text else default or ""
            return default or ""
        except Exception as e:
            logger.error(f"Error extracting text for tag {tag_name}: {e}")
            return default or ""

    async def process_documents(self) -> AsyncGenerator[DomainDocument, None]:
        """Process HMDB XML file as a stream of documents."""
        # Original method implementation

    async def process_batch(
        self, batch_size: int = 100
    ) -> AsyncGenerator[List[Any], None]:
        """Process data in batches - delegates to process_metabolite_batch."""
        async for batch in self.process_metabolite_batch(batch_size):
            yield batch

    async def process_metabolite_batch(
        self, batch_size: int = 100
    ) -> AsyncGenerator[List[dict], None]:
        """Process HMDB XML file in batches yielding raw metabolite dictionaries.

        Memory-efficient streaming parser that doesn't load the entire XML into memory.

        Args:
            batch_size: Number of metabolites to process in each batch

        Yields:
            List of metabolite dictionaries with extracted data
        """
        batch: List[dict] = []
        processed = 0

        try:
            # Use iterparse for memory-efficient streaming
            context = ET.iterparse(self.xml_file, events=("start", "end"))
            context = iter(context)
            event, root = next(context)

            # Use indeterminate progress bar if total is unknown
            pbar = tqdm(
                total=self._total_compounds if self._total_compounds > 0 else None,
                desc="Processing compounds for Qdrant",
            )

            for event, elem in context:
                # Handle namespace - check if tag ends with 'metabolite'
                if event == "end" and elem.tag.endswith("metabolite"):
                    try:
                        compound = elem

                        # Get all synonyms including IUPAC names (handling namespace)
                        synonyms = [
                            cast(str, syn.text)
                            for syn in compound.findall(".//*")
                            if syn is not None
                            and syn.tag.endswith("synonym")
                            and syn.text
                            and syn.text.strip()
                        ]

                        # Add IUPAC names to synonyms if they exist and aren't already included
                        iupac = self._get_text(compound, "iupac_name")
                        trad_iupac = self._get_text(compound, "traditional_iupac")
                        if iupac and iupac not in synonyms:
                            synonyms.append(iupac)
                        if trad_iupac and trad_iupac not in synonyms:
                            synonyms.append(trad_iupac)

                        # Create metabolite dictionary with all relevant fields
                        metabolite_dict = {
                            "hmdb_id": self._get_text(compound, "accession"),
                            "name": self._get_text(compound, "name"),
                            "description": self._get_text(compound, "description"),
                            "chemical_formula": self._get_text(
                                compound, "chemical_formula"
                            ),
                            "iupac_name": iupac,
                            "traditional_iupac": trad_iupac,
                            "synonyms": synonyms,
                            "inchikey": self._get_text(compound, "inchikey"),
                            "cas_registry_number": self._get_text(
                                compound, "cas_registry_number"
                            ),
                            "kegg_id": self._get_text(compound, "kegg_id"),
                            "pubchem_compound_id": self._get_text(
                                compound, "pubchem_compound_id"
                            ),
                            "chebi_id": self._get_text(compound, "chebi_id"),
                            "drugbank_id": self._get_text(compound, "drugbank_id"),
                            "foodb_id": self._get_text(compound, "foodb_id"),
                            "hmdb_id_alt": self._get_text(
                                compound, "secondary_accessions"
                            ),
                            "molecular_weight": self._get_text(
                                compound, "molecular_weight"
                            ),
                            "monoisotopic_molecular_weight": self._get_text(
                                compound, "monoisotopic_molecular_weight"
                            ),
                            "smiles": self._get_text(compound, "smiles"),
                            "inchi": self._get_text(compound, "inchi"),
                        }

                        # Only add if we have required fields
                        if metabolite_dict["hmdb_id"] and metabolite_dict["name"]:
                            batch.append(metabolite_dict)
                            processed += 1

                        if len(batch) >= batch_size:
                            yield batch
                            batch = []

                    except Exception as e:
                        logger.error(f"Error processing compound: {e}")
                    finally:
                        # Clear the element to save memory
                        elem.clear()
                        # For ElementTree, we just clear the element
                        # (getprevious is lxml-specific)
                        pbar.update(1)

            pbar.close()

            # Yield any remaining items
            if batch:
                yield batch

        except Exception as e:
            logger.error(f"Error processing XML file: {e}")
            raise

        finally:
            logger.info(f"Processed {processed} metabolite dictionaries")
