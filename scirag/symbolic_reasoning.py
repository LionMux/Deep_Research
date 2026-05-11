"""
SciRAG Symbolic Reasoning — 3-step pipeline for paper relationship understanding.

Step 1: Segment extraction — extract T/E/A segments + SPO triples + info-units from each paper
Step 2: Relationship building — build symbolic links [1]T → [2]E → [Q] with triple evidence
Step 3: Selection & ranking — select papers with 3-sentence reasoning

Inspired by:
  - Yale NLP SciRAG ACL 2025
  - Liu-Hy/nlp-contrib-graph (SemEval-2021 Task 11 Winner)
    Key functions integrated: heading detection, entity spans, SPO triples,
    info-unit classification, cross-sentence resolution
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

from .llm_client import KimiClient
from .config import SciRAGConfig
from .tree_node import TreeNode

logger = logging.getLogger(__name__)

# ========================================================================
# nlp-contrib-graph integration: constants and heuristics
# (From Liu-Hy/nlp-contrib-graph, SemEval-2021 Task 11 Winner)
# ========================================================================

# Info-unit types from nlp-contrib-graph
INFO_UNIT_TYPES = {
    "Contribution": "What the paper contributes (method, result, finding)",
    "Task": "The problem/task the paper addresses",
    "Definition": "Definitions of key concepts",
    "Support": "Evidence or support for claims",
    "Positioning": "How the work relates to prior work",
    "Results": "Experimental results and metrics",
}

# SPO (Subject-Predicate-Object) phrase types from nlp-contrib-graph
SPO_TYPES = {
    "s": "Subject — the entity performing the action or being described",
    "p": "Predicate — the action or relationship verb/phrase",
    "ob": "Object — the entity affected by the action or the target",
    "b": "Both subject and object (shared entity)",
}

# good_state: boolean indicator → SPO type mapping from nlp-contrib-graph/ext.py:405
# Each key maps to [is_subject, is_predicate, is_object]
SPO_GOOD_STATES = {
    "p": [0, 1, 0],   # Predicate only
    "s": [1, 0, 0],   # Subject only
    "ob": [0, 0, 1],  # Object only
    "b": [1, 0, 1],   # Both subject and object
}


def is_heading(line: str, main=False) -> bool:
    """
    Detect if a line is a section heading.
    From nlp-contrib-graph/ext.py (adapted).
    Heuristics: short, numbered, ALL CAPS, or Title Case.
    """
    line = line.strip()
    if not line or len(line) > 100:
        return False
    # Numbered heading: "1. Introduction", "2.3 Methods", "1)", "I."
    if re.match(r'^(\d+(\.\d+)*[\.\)]?\s+\w|[IVX]+[\.\)]\s+\w)', line):
        return True
    # ALL CAPS heading (2+ words)
    if line.isupper() and len(line.split()) >= 2:
        return True
    # Title Case heading (2+ words, no ending punctuation)
    words = line.split()
    if len(words) >= 2 and words[0][0].isupper():
        capitalized = sum(1 for w in words if w and w[0].isupper())
        if capitalized >= max(len(words) * 0.5, 1):
            if not line[-1] in '.,;:!?':
                return True
    # Short heading (1-3 words, ALL CAPS or Title Case, no ending punctuation)
    if len(words) <= 3 and len(line) > 3:
        if not line[-1] in '.,;:!?':
            if line.isupper() or (words[0][0].isupper() if words else False):
                return True
    return False


def is_main_heading(line: str) -> bool:
    """Detect main (top-level) section headings."""
    MAIN_HEADING_KEYWORDS = [
        'abstract', 'introduction', 'related work', 'background',
        'method', 'approach', 'model', 'architecture',
        'experiment', 'evaluation', 'results', 'analysis',
        'discussion', 'conclusion', 'future work',
        'theorem', 'proof', 'lemma', 'definition',
        'implementation', 'application', 'dataset',
    ]
    line_lower = line.strip().lower()
    for keyword in MAIN_HEADING_KEYWORDS:
        if keyword in line_lower and is_heading(line):
            return True
    if re.match(r'^(\d+[\.\)]\s|[IVX]+[\.\)]\s)', line.strip()):
        return True
    return False


def split_into_paragraphs(text: str) -> List[Tuple[str, str]]:
    """Split text into (heading, paragraph) tuples."""
    paragraphs = []
    lines = text.split('\n')
    current_heading = ""
    current_paragraph = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if is_heading(stripped):
            if current_paragraph:
                paragraphs.append((current_heading, '\n'.join(current_paragraph)))
                current_paragraph = []
            current_heading = stripped
        else:
            current_paragraph.append(stripped)
    if current_paragraph:
        paragraphs.append((current_heading, '\n'.join(current_paragraph)))
    return paragraphs


# ========================================================================
# Data structures
# ========================================================================

@dataclass
class Segment:
    """One T/E/A segment from a paper."""
    paper_id: str
    paper_title: str
    tag: str  # T, E, M, A
    text: str
    section_name: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "paper": self.paper_id,
            "tag": self.tag,
            "text": self.text[:200],
            "section": self.section_name,
        }


@dataclass
class Triple:
    """
    SPO triple extracted from text.
    From nlp-contrib-graph: triples are stored as "Subject||Predicate||Object".
    """
    subject: str
    predicate: str
    object: str
    paper_id: str = ""
    sentence_idx: int = 0
    info_unit: str = ""  # Which info-unit this triple belongs to
    # Completeness tracking (from nlp-contrib-graph):
    # 3 = all phrases from single sentence (triple_A)
    # 2 = 2 phrases from single sentence (triple_B/C)
    # 1 = 1 phrase from single sentence (triple_D)
    # 0 = cross-sentence (requires resolution)
    completeness: int = 0

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "paper_id": self.paper_id,
            "info_unit": self.info_unit,
            "completeness": self.completeness,
        }

    def __hash__(self):
        return hash((self.subject, self.predicate, self.object))

    def __eq__(self, other):
        if isinstance(other, Triple):
            return (self.subject == other.subject and
                    self.predicate == other.predicate and
                    self.object == other.object)
        return False


@dataclass
class InfoUnit:
    """
    A sentence/paragraph classified into an info-unit type.
    From nlp-contrib-graph info-unit classification.
    """
    text: str
    unit_type: str  # Contribution, Task, Definition, Support, Positioning, Results
    paper_id: str = ""
    sentence_idx: int = 0
    main_heading: str = ""   # Main section heading (from nlp-contrib-graph)
    sub_heading: str = ""    # Sub-section heading
    # BIO tag sequence for entity spans (from nlp-contrib-graph)
    bio_tags: Optional[List[str]] = None
    # Extracted predicates from this sentence
    predicates: List[Tuple[str, Tuple[int, int]]] = field(default_factory=list)
    # Extracted subjects/objects
    entities: List[Tuple[str, Tuple[int, int]]] = field(default_factory=list)
    # Triples fully contained in this sentence (completeness=3)
    triples_full: List[Triple] = field(default_factory=list)
    # Triples partially contained (completeness=2)
    triples_partial: List[Triple] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text[:200],
            "unit_type": self.unit_type,
            "paper_id": self.paper_id,
            "main_heading": self.main_heading,
            "predicates": [p[0] for p in self.predicates],
            "entities": [e[0] for e in self.entities],
            "triples_count": len(self.triples_full) + len(self.triples_partial),
        }


@dataclass
class SymbolicRelationship:
    """Relationship between two segments, supported by triples."""
    from_paper: str
    from_tag: str
    to_paper: str
    to_tag: str
    relation: str  # "supports", "extends", "contradicts", "applies"
    explanation: str = ""
    # Evidence triples supporting this relationship
    evidence_triples: List[Triple] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "from": f"[{self.from_paper}]{self.from_tag}",
            "to": f"[{self.to_paper}]{self.to_tag}",
            "relation": self.relation,
            "explanation": self.explanation[:100],
            "evidence_triples": [t.to_dict() for t in self.evidence_triples[:3]],
        }


# ========================================================================
# SymbolicReasoner — full pipeline
# ========================================================================

class SymbolicReasoner:
    """
    3-step symbolic reasoning pipeline with nlp-contrib-graph integration.

    Step 1: Extract T/E/A segments + SPO triples + info-units from each paper
    Step 2: Build relationships between segments (with triple evidence)
    Step 3: Select & rank papers for each outline section
    """

    def __init__(self, client: KimiClient, config: SciRAGConfig):
        self.client = client
        self.config = config

    # ========================================================================
    # STEP 1a: T/E/A Segment Extraction
    # ========================================================================

    def extract_segments(self, paper_id: str, paper_title: str, text: str) -> List[Segment]:
        """
        Extract T/E/A segments from a paper.
        Returns list of segments with tags.
        """
        prompt = f"""Analyze this academic paper and extract segments by type.

Paper: {paper_title} ({paper_id})

Text:
{text[:3000]}

Extract segments in JSON:
{{
  "segments": [
    {{"tag": "T", "text": "theoretical claim or theorem", "section": "section name"}},
    {{"tag": "E", "text": "experimental result or evaluation", "section": "section name"}},
    {{"tag": "M", "text": "method or algorithm description", "section": "section name"}},
    {{"tag": "A", "text": "application or deployment", "section": "section name"}}
  ]
}}

T = Theory (theorems, proofs, frameworks, bounds)
E = Experiment (benchmarks, metrics, ablations)
M = Method (algorithms, architectures, techniques)
A = Application (real-world use, systems, products)
"""

        messages = [
            {"role": "system", "content": "You are an academic paper analyzer. Extract structured segments by type."},
            {"role": "user", "content": prompt},
        ]

        result = self.client.chat_json(
            messages,
            model=self.config.kimi_model_outline,
            temperature=0.1,
            max_tokens=2048,
        )

        segments = []
        for s in result.get("segments", []):
            segments.append(Segment(
                paper_id=paper_id,
                paper_title=paper_title,
                tag=s.get("tag", "M"),
                text=s.get("text", ""),
                section_name=s.get("section", ""),
            ))

        logger.debug(f"Extracted {len(segments)} segments from {paper_id}")
        return segments

    def extract_segments_batch(self, papers: List[Dict]) -> Dict[str, List[Segment]]:
        """
        Extract segments from multiple papers.
        Returns {{paper_id: [segments]}}.
        """
        results = {}
        for paper in papers:
            pid = paper.get("id", "")
            if not pid:
                continue
            segments = self.extract_segments(
                paper_id=pid,
                paper_title=paper.get("title", ""),
                text=paper.get("text", ""),
            )
            results[pid] = segments
        return results

    # ========================================================================
    # STEP 1b: SPO Triple Extraction (from nlp-contrib-graph)
    # ========================================================================

    def extract_triples(self, paper_id: str, paper_title: str, text: str) -> List[Triple]:
        """
        Extract SPO (Subject-Predicate-Object) triples from paper text.

        Adapted from nlp-contrib-graph/ext.py triple extraction logic.
        Uses LLM to extract triples in the format "Subject||Predicate||Object".

        Args:
            paper_id: Unique paper identifier
            paper_title: Paper title
            text: Full paper text

        Returns:
            List of Triple objects with completeness scoring
        """
        # First split into paragraphs with headings
        paragraphs = split_into_paragraphs(text)

        all_triples = []
        sentence_idx = 0

        for heading, para_text in paragraphs:
            if not para_text.strip():
                continue

            # Process each paragraph for triples
            para_triples = self._extract_triples_from_paragraph(
                paper_id=paper_id,
                paper_title=paper_title,
                heading=heading,
                text=para_text,
                base_sentence_idx=sentence_idx,
            )
            all_triples.extend(para_triples)
            sentence_idx += len(para_text.split('. '))

        logger.debug(f"Extracted {len(all_triples)} triples from {paper_id}")
        return all_triples

    def _extract_triples_from_paragraph(
        self, paper_id: str, paper_title: str,
        heading: str, text: str, base_sentence_idx: int = 0
    ) -> List[Triple]:
        """
        Extract triples from a single paragraph using LLM.

        Prompts LLM to extract SPO triples following nlp-contrib-graph format.
        Each triple: Subject||Predicate||Object
        """
        prompt = f"""Extract Subject-Predicate-Object triples from this academic text.

Paper: {paper_title}
Section: {heading or "Unknown"}

Text:
{text[:1500]}

Extract triples that capture key factual relationships.
Format each triple as: Subject||Predicate||Object

Respond in JSON:
{{
  "triples": [
    {{
      "subject": "neural network",
      "predicate": "achieves",
      "object": "95% accuracy",
      "info_unit": "Results"
    }},
    {{
      "subject": "BERT",
      "predicate": "uses",
      "object": "transformer architecture",
      "info_unit": "Contribution"
    }}
  ]
}}

Guidelines:
- Subject: the main entity (model, method, concept, paper)
- Predicate: the relationship/action verb (achieves, uses, proposes, outperforms)
- Object: the target/result (metric, technique, finding)
- info_unit: one of [Contribution, Task, Definition, Support, Positioning, Results]
- Only extract substantive, verifiable triples
- Skip vague or meta-discursive statements
"""

        messages = [
            {"role": "system", "content": "You extract structured SPO triples from academic text. Be precise and factual."},
            {"role": "user", "content": prompt},
        ]

        try:
            result = self.client.chat_json(
                messages,
                model=self.config.kimi_model_outline,
                temperature=0.1,
                max_tokens=2048,
            )
        except Exception as e:
            logger.warning(f"Triple extraction LLM call failed: {e}")
            return []

        triples = []
        for i, t in enumerate(result.get("triples", [])):
            subject = t.get("subject", "").strip()
            predicate = t.get("predicate", "").strip()
            obj = t.get("object", "").strip()

            if not subject or not predicate or not obj:
                continue

            triples.append(Triple(
                subject=subject,
                predicate=predicate,
                object=obj,
                paper_id=paper_id,
                sentence_idx=base_sentence_idx + i,
                info_unit=t.get("info_unit", ""),
                completeness=3,  # LLM extraction assumes all from same context
            ))

        return triples

    def extract_triples_batch(self, papers: List[Dict]) -> Dict[str, List[Triple]]:
        """
        Extract triples from multiple papers.
        Returns {{paper_id: [triples]}}.
        """
        results = {}
        for paper in papers:
            pid = paper.get("id", "")
            if not pid:
                continue
            triples = self.extract_triples(
                paper_id=pid,
                paper_title=paper.get("title", ""),
                text=paper.get("text", ""),
            )
            results[pid] = triples
        return results

    # ========================================================================
    # STEP 1c: Info-Unit Classification (from nlp-contrib-graph)
    # ========================================================================

    def classify_info_units(self, paper_id: str, paper_title: str, text: str) -> List[InfoUnit]:
        """
        Classify sentences/paragraphs into info-unit types.

        Adapted from nlp-contrib-graph/ext.py info-unit assignment logic.
        In the original, info-units are assigned by matching source sentences
        from JSON annotation files. Here we use LLM for automatic classification.

        Args:
            paper_id: Unique paper identifier
            paper_title: Paper title
            text: Full paper text

        Returns:
            List of InfoUnit objects with type assignments
        """
        paragraphs = split_into_paragraphs(text)
        info_units = []

        for heading, para_text in paragraphs:
            if not para_text.strip():
                continue

            units = self._classify_paragraph_info_units(
                paper_id=paper_id,
                paper_title=paper_title,
                heading=heading,
                text=para_text,
            )
            info_units.extend(units)

        logger.debug(f"Classified {len(info_units)} info-units from {paper_id}")
        return info_units

    def _classify_paragraph_info_units(
        self, paper_id: str, paper_title: str,
        heading: str, text: str
    ) -> List[InfoUnit]:
        """
        Classify sentences in a paragraph into info-unit types using LLM.

        Follows nlp-contrib-graph's 6 info-unit types:
        Contribution, Task, Definition, Support, Positioning, Results
        """
        # Split into sentences
        sentences = [s.strip() for s in text.split('. ') if s.strip() and len(s.strip()) > 10]
        if not sentences:
            return []

        prompt = f"""Classify each sentence of this academic text into info-unit types.

Paper: {paper_title}
Section: {heading or "Unknown"}

Sentences:
{chr(10).join(f"{i+1}. {s[:200]}" for i, s in enumerate(sentences[:10]))}

Info-unit types:
- Contribution: what the paper contributes (method, result, finding)
- Task: the problem/task addressed
- Definition: definitions of key concepts
- Support: evidence or support for claims
- Positioning: how the work relates to prior work
- Results: experimental results and metrics

Respond in JSON:
{{
  "classifications": [
    {{"sentence_idx": 1, "unit_type": "Contribution", "key_phrases": ["phrase1", "phrase2"]}},
    {{"sentence_idx": 2, "unit_type": "Results", "key_phrases": ["95% accuracy"]}}
  ]
}}

Also extract key phrases (predicates and entities) for each sentence.
"""

        messages = [
            {"role": "system", "content": "You classify academic sentences into info-unit types and extract key phrases."},
            {"role": "user", "content": prompt},
        ]

        try:
            result = self.client.chat_json(
                messages,
                model=self.config.kimi_model_outline,
                temperature=0.1,
                max_tokens=2048,
            )
        except Exception as e:
            logger.warning(f"Info-unit classification failed: {e}")
            # Fallback: assign all as generic
            return [InfoUnit(
                text=text[:500],
                unit_type="Contribution",
                paper_id=paper_id,
                main_heading=heading,
            )]

        units = []
        for c in result.get("classifications", []):
            idx = c.get("sentence_idx", 1) - 1
            if 0 <= idx < len(sentences):
                sentence = sentences[idx]
            else:
                sentence = text[:300]

            unit_type = c.get("unit_type", "Contribution")
            # Validate unit_type
            if unit_type not in INFO_UNIT_TYPES:
                unit_type = "Contribution"

            # Extract key phrases
            key_phrases = c.get("key_phrases", [])
            predicates = []
            entities = []
            for phrase in key_phrases:
                # Simple heuristic: verbs are predicates, nouns are entities
                if any(v in phrase.lower() for v in ['achieve', 'use', 'propose', 'show', 'demonstrate', 'outperform', 'improve', 'reduce', 'increase', 'employ', 'adopt', 'apply']):
                    predicates.append((phrase, (0, 0)))
                else:
                    entities.append((phrase, (0, 0)))

            units.append(InfoUnit(
                text=sentence,
                unit_type=unit_type,
                paper_id=paper_id,
                main_heading=heading,
                predicates=predicates,
                entities=entities,
            ))

        return units

    def classify_info_units_batch(self, papers: List[Dict]) -> Dict[str, List[InfoUnit]]:
        """
        Classify info-units for multiple papers.
        Returns {{paper_id: [info_units]}}.
        """
        results = {}
        for paper in papers:
            pid = paper.get("id", "")
            if not pid:
                continue
            units = self.classify_info_units(
                paper_id=pid,
                paper_title=paper.get("title", ""),
                text=paper.get("text", ""),
            )
            results[pid] = units
        return results

    # ========================================================================
    # STEP 1d: Cross-Sentence Triple Resolution (from nlp-contrib-graph)
    # ========================================================================

    def resolve_cross_sentence(self, triples: List[Triple],
                               info_units: Dict[str, List[InfoUnit]]) -> List[Triple]:
        """
        Resolve triples that span multiple sentences.

        Adapted from nlp-contrib-graph/ext.py cross-sentence resolution.
        In the original, ~X% of triples cannot get all phrases from a single sentence.
        This method merges partial triples across sentence boundaries.

        Algorithm:
        1. Group triples by paper_id
        2. For triples with completeness < 3, find matching phrases in other sentences
        3. Merge partial matches into complete triples
        4. Return all triples with updated completeness scores

        Args:
            triples: All extracted triples
            info_units: {paper_id: [InfoUnit]} for context

        Returns:
            Triples with cross-sentence resolution applied
        """
        # Group by paper
        by_paper: Dict[str, List[Triple]] = {}
        for t in triples:
            by_paper.setdefault(t.paper_id, []).append(t)

        resolved = []
        for paper_id, paper_triples in by_paper.items():
            paper_units = info_units.get(paper_id, [])

            # Build entity index from info-units
            entity_index: Dict[str, List[int]] = {}  # entity -> [sentence indices]
            for i, unit in enumerate(paper_units):
                for entity_text, _ in unit.entities:
                    entity_index.setdefault(entity_text.lower(), []).append(i)
                # Also index words from the text
                for word in unit.text.lower().split():
                    if len(word) > 3:
                        entity_index.setdefault(word, []).append(i)

            # Try to resolve incomplete triples
            for triple in paper_triples:
                if triple.completeness >= 3:
                    resolved.append(triple)
                    continue

                # Try to find matching entities across sentences
                subj_matches = entity_index.get(triple.subject.lower(), [])
                pred_matches = entity_index.get(triple.predicate.lower(), [])
                obj_matches = entity_index.get(triple.object.lower(), [])

                all_matches = set(subj_matches + pred_matches + obj_matches)

                if len(all_matches) >= 2:
                    # Cross-sentence resolution: phrases found in multiple sentences
                    triple.completeness = 2  # 2 phrases from related sentences
                    resolved.append(triple)
                elif len(all_matches) == 1:
                    triple.completeness = 1  # 1 phrase found
                    resolved.append(triple)
                else:
                    # No matches — keep as cross-sentence with completeness 0
                    triple.completeness = 0
                    resolved.append(triple)

        # Stats
        completeness_counts = {}
        for t in resolved:
            completeness_counts[t.completeness] = completeness_counts.get(t.completeness, 0) + 1

        logger.info(f"Cross-sentence resolution: { {k: v for k, v in sorted(completeness_counts.items())} }")
        return resolved

    def _get_entity_spans(self, bio_tags: List[str]) -> List[Tuple[int, int]]:
        """
        Extract entity spans from BIO tag sequence.
        From nlp-contrib-graph/ext.py get_entity_spans().

        Args:
            bio_tags: List of BIO tags ['O', 'B', 'I', 'O', ...]

        Returns:
            List of (start, end) tuples for each entity span
        """
        spans = []
        for i in range(len(bio_tags)):
            if bio_tags[i] == 'B':
                st, ed = i, i + 1
                for j in range(i + 1, len(bio_tags)):
                    if bio_tags[j] == 'I':
                        ed += 1
                    else:
                        break
                spans.append((st, ed))
        return spans

    # ========================================================================
    # STEP 2: Relationship Building (enhanced with triple evidence)
    # ========================================================================

    def build_relationships(self, segments: Dict[str, List[Segment]],
                           triples: Dict[str, List[Triple]],
                           query: str) -> List[SymbolicRelationship]:
        """
        Build symbolic relationships between paper segments.
        Enhanced with SPO triple evidence from nlp-contrib-graph.

        Example: [Karpukhin2020]T (dual-encoder theory) → [Xiong2021]E (ANCE experiment)
        Supported by triples: "dual-encoder||enables||dense retrieval"
        """
        # Flatten all segments
        all_segments = []
        for pid, segs in segments.items():
            all_segments.extend(segs)

        if len(all_segments) < 2:
            return []

        # Build prompt with segments and triples
        segments_text = "\n".join(
            f"[{s.paper_id}]{s.tag}: {s.text[:150]}"
            for s in all_segments[:20]
        )

        # Include triples as evidence
        all_triples = []
        for pid, trip_list in triples.items():
            all_triples.extend(trip_list)

        triples_text = "\n".join(
            f"  [{t.paper_id}] {t.subject}||{t.predicate}||{t.object} ({t.info_unit})"
            for t in all_triples[:30]
        )

        prompt = f"""Analyze relationships between these paper segments for query: "{query}"

Segments:
{segments_text}

Supporting triples (SPO):
{triples_text}

Identify relationships in JSON:
{{
  "relationships": [
    {{
      "from": "[PaperID]T",
      "to": "[PaperID]E",
      "relation": "supports",
      "explanation": "Theory from paper 1 supports experiment in paper 2",
      "evidence_triples": ["triple subject||predicate||object"]
    }}
  ]
}}

Relations: supports, extends, contradicts, applies, improves_upon
Use the SPO triples as evidence for each relationship.
"""

        messages = [
            {"role": "system", "content": "You analyze relationships between academic paper segments, using SPO triples as evidence."},
            {"role": "user", "content": prompt},
        ]

        result = self.client.chat_json(
            messages,
            model=self.config.kimi_model_outline,
            temperature=0.1,
            max_tokens=2048,
        )

        relationships = []
        # Build triple lookup for evidence matching
        triple_lookup = {}
        for t in all_triples:
            key = f"{t.subject}||{t.predicate}||{t.object}"
            triple_lookup[key.lower()] = t

        for r in result.get("relationships", []):
            from_str = r.get("from", "")
            to_str = r.get("to", "")
            from_paper, from_tag = self._parse_paper_tag(from_str)
            to_paper, to_tag = self._parse_paper_tag(to_str)

            # Match evidence triples
            evidence_triples = []
            for et in r.get("evidence_triples", []):
                et_lower = et.lower()
                if et_lower in triple_lookup:
                    evidence_triples.append(triple_lookup[et_lower])

            relationships.append(SymbolicRelationship(
                from_paper=from_paper,
                from_tag=from_tag,
                to_paper=to_paper,
                to_tag=to_tag,
                relation=r.get("relation", "supports"),
                explanation=r.get("explanation", ""),
                evidence_triples=evidence_triples,
            ))

        logger.info(f"Built {len(relationships)} symbolic relationships with triple evidence")
        return relationships

    def _parse_paper_tag(self, s: str) -> Tuple[str, str]:
        """Parse '[PaperID]T' → (PaperID, 'T')."""
        m = re.match(r'\[(.*?)\]([TEMA])', s)
        if m:
            return m.group(1), m.group(2)
        return s, "M"

    # ========================================================================
    # STEP 3: Selection & Ranking (enhanced with info-unit coverage)
    # ========================================================================

    def select_papers(self, segments: Dict[str, List[Segment]],
                     triples: Dict[str, List[Triple]],
                     info_units: Dict[str, List[InfoUnit]],
                     relationships: List[SymbolicRelationship],
                     outline_section: TreeNode,
                     top_k: int = 5) -> List[Dict]:
        """
        Select and rank papers for an outline section.
        Enhanced with info-unit coverage analysis from nlp-contrib-graph.

        Returns papers with 3-sentence reasoning and info-unit coverage stats.
        """
        # Build context
        segments_text = "\n".join(
            f"[{pid}]{s.tag}: {s.text[:100]}"
            for pid, segs in segments.items()
            for s in segs[:3]
        )

        # Include info-unit coverage
        unit_coverage = {}
        for pid, units in info_units.items():
            type_counts = {}
            for u in units:
                type_counts[u.unit_type] = type_counts.get(u.unit_type, 0) + 1
            unit_coverage[pid] = type_counts

        coverage_text = "\n".join(
            f"  [{pid}] {', '.join(f'{t}:{n}' for t, n in types.items())}"
            for pid, types in unit_coverage.items()
        )

        rel_text = "\n".join(
            f"{r.from_paper}{r.from_tag} → {r.relation} → {r.to_paper}{r.to_tag}: {r.explanation[:80]}"
            for r in relationships[:10]
        )

        prompt = f"""Select the most relevant papers for this section:

Section: {outline_section.title}
Description: {outline_section.description}
Keywords: {', '.join(outline_section.keywords)}

Available segments:
{segments_text}

Info-unit coverage per paper:
{coverage_text}

Relationships:
{rel_text}

Respond in JSON:
{{
  "selected": [
    {{
      "paper_id": "...",
      "relevance_score": 0.95,
      "reasoning": "1. Paper X provides the theoretical foundation. 2. Its method directly addresses the query. 3. Experimental results confirm applicability.",
      "info_unit_coverage": ["Contribution", "Results"]
    }}
  ]
}}

Provide 3-sentence reasoning for each selection.
Consider info-unit coverage (Contribution, Task, Results, etc.) in your ranking.
"""

        messages = [
            {"role": "system", "content": "You select and rank academic papers with detailed reasoning, considering info-unit coverage."},
            {"role": "user", "content": prompt},
        ]

        result = self.client.chat_json(
            messages,
            model=self.config.kimi_model_outline,
            temperature=0.1,
            max_tokens=2048,
        )

        selected = result.get("selected", [])
        logger.info(f"Selected {len(selected)} papers for '{outline_section.title}'")
        return selected

    # ========================================================================
    # Utility: Find source sentences (from nlp-contrib-graph/parse.py)
    # ========================================================================

    @staticmethod
    def find_source(data: dict, ls: Optional[list] = None) -> list:
        """
        Recursively find all source sentences in a nested dict/list structure.
        From nlp-contrib-graph/parse.py find_source().

        Used to extract source sentences from info-unit JSON annotations.
        """
        if ls is None:
            ls = []
        if isinstance(data, dict):
            for key in data.keys():
                if key == "from sentence":
                    sentences = data[key].split('\n')
                    for s in sentences:
                        ls.append(s.strip())
                elif isinstance(data[key], dict):
                    SymbolicReasoner.find_source(data[key], ls)
                elif isinstance(data[key], list):
                    for i in data[key]:
                        SymbolicReasoner.find_source(i, ls)
        elif isinstance(data, list):
            for i in data:
                SymbolicReasoner.find_source(i, ls)
        return ls

    @staticmethod
    def find_tri_sent(data, triple: List[str], trace: Optional[list] = None,
                      ls: Optional[list] = None, prefix: Optional[list] = None) -> list:
        """
        Recursively find source sentences that contain a given triple.
        From nlp-contrib-graph/parse.py find_tri_sent().

        Traverses nested dict/list structure to match triple phrases.
        """
        if trace is None:
            trace = []
        if ls is None:
            ls = []
        if prefix is None:
            prefix = []

        if isinstance(data, dict):
            for i, key in enumerate(data.keys()):
                if key != "from sentence":
                    if prefix and i != 0:
                        trace += prefix
                    trace.append(key)
                    SymbolicReasoner.find_tri_sent(
                        data[key], triple, trace, ls,
                        SymbolicReasoner._get_prefix(data[key], trace)
                    )
                else:
                    if SymbolicReasoner._is_contained(trace, triple):
                        ls.append(data[key].strip())
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if prefix and i != 0:
                    trace += prefix
                SymbolicReasoner.find_tri_sent(item, triple, trace, ls, prefix)
        elif isinstance(data, str):
            trace.append(data)
        return ls

    @staticmethod
    def _is_contained(trace: list, triple: List[str]) -> bool:
        """Check if a triple is contained in a trace path. From nlp-contrib-graph/parse.py."""
        if len(trace) >= len(triple):
            for i in range(len(trace) - 2):
                if trace[i:i+3] == triple:
                    return True
            return False
        return False

    @staticmethod
    def _get_prefix(data, trace: list) -> list:
        """Get prefix for coordinated items. From nlp-contrib-graph/parse.py."""
        if isinstance(data, dict) or isinstance(data, list):
            return trace[-2:]
        return []

    # ========================================================================
    # Full Pipeline
    # ========================================================================

    def run(self, papers: List[Dict], query: str, outline: TreeNode) -> Dict:
        """
        Run full 3-step symbolic reasoning pipeline with nlp-contrib-graph integration.

        Phase 2 additions:
          - SPO triple extraction (Step 1b)
          - Info-unit classification (Step 1c)
          - Cross-sentence resolution (Step 1d)
          - Triple-enhanced relationship building (Step 2)
          - Info-unit-aware selection (Step 3)

        Returns:
            {{
                "segments": {{paper_id: [Segment]}},
                "triples": {{paper_id: [Triple]}},
                "info_units": {{paper_id: [InfoUnit]}},
                "relationships": [SymbolicRelationship],
                "selections": {{section_title: [selected_papers]}},
                "stats": {{...}},
            }}
        """
        logger.info("=" * 50)
        logger.info("Starting Symbolic Reasoning Pipeline (Phase 2)")
        logger.info("=" * 50)

        # Step 1a: Extract T/E/A segments
        logger.info("[Symbolic] Step 1a: Extracting T/E/A segments...")
        segments = self.extract_segments_batch(papers)
        total_segments = sum(len(s) for s in segments.values())
        logger.info(f"[Symbolic] Extracted {total_segments} segments from {len(papers)} papers")

        # Step 1b: Extract SPO triples (from nlp-contrib-graph)
        logger.info("[Symbolic] Step 1b: Extracting SPO triples...")
        triples = self.extract_triples_batch(papers)
        total_triples = sum(len(t) for t in triples.values())
        logger.info(f"[Symbolic] Extracted {total_triples} triples")

        # Step 1c: Classify info-units (from nlp-contrib-graph)
        logger.info("[Symbolic] Step 1c: Classifying info-units...")
        info_units = self.classify_info_units_batch(papers)
        total_units = sum(len(u) for u in info_units.values())
        logger.info(f"[Symbolic] Classified {total_units} info-units")

        # Step 1d: Cross-sentence resolution (from nlp-contrib-graph)
        logger.info("[Symbolic] Step 1d: Cross-sentence triple resolution...")
        all_triples = []
        for tlist in triples.values():
            all_triples.extend(tlist)
        resolved_triples = self.resolve_cross_sentence(all_triples, info_units)
        # Update triples dict with resolved
        triples_resolved: Dict[str, List[Triple]] = {}
        for t in resolved_triples:
            triples_resolved.setdefault(t.paper_id, []).append(t)
        fully_resolved = sum(1 for t in resolved_triples if t.completeness >= 2)
        logger.info(f"[Symbolic] Cross-sentence: {fully_resolved}/{len(resolved_triples)} triples fully resolved")

        # Step 2: Build relationships with triple evidence
        logger.info("[Symbolic] Step 2: Building relationships...")
        relationships = self.build_relationships(segments, triples_resolved, query)

        # Step 3: Select papers per section with info-unit awareness
        logger.info("[Symbolic] Step 3: Selecting papers per section...")
        selections = {}
        for child in outline.children:
            selected = self.select_papers(
                segments, triples_resolved, info_units, relationships, child
            )
            selections[child.title] = selected

        logger.info("Symbolic reasoning complete")

        # Compute info-unit distribution
        unit_type_counts: Dict[str, int] = {}
        for units in info_units.values():
            for u in units:
                unit_type_counts[u.unit_type] = unit_type_counts.get(u.unit_type, 0) + 1

        return {
            "segments": {pid: [s.to_dict() for s in segs] for pid, segs in segments.items()},
            "triples": {pid: [t.to_dict() for t in ts] for pid, ts in triples_resolved.items()},
            "info_units": {pid: [u.to_dict() for u in us] for pid, us in info_units.items()},
            "relationships": [r.to_dict() for r in relationships],
            "selections": selections,
            "stats": {
                "papers": len(papers),
                "total_segments": total_segments,
                "total_triples": len(resolved_triples),
                "fully_resolved_triples": fully_resolved,
                "total_info_units": total_units,
                "info_unit_distribution": unit_type_counts,
                "relationships": len(relationships),
                "sections": len(outline.children),
            },
        }
