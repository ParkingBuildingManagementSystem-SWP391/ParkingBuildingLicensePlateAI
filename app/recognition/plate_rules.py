from __future__ import annotations

import re
from dataclasses import dataclass


VALID_PROVINCE_CODES = {
    "11", "12", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26",
    "27", "28", "29", "30", "31", "32", "33", "34", "36", "37", "38", "40", "41", "43", "47",
    "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62",
    "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77",
    "78", "79", "80", "81", "82", "83", "84", "85", "86", "88", "89", "90", "92", "93", "94",
    "95", "97", "98", "99",
}
SERIES_LETTERS = set("ABCDEFGHKLMNPSTUVXYZ")
SPECIAL_SERIES = {"LD", "DA", "NG", "QT", "MD"}
CHAR_TO_DIGIT = {
    "O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "T": "1",
    "Z": "2", "E": "3", "A": "4", "S": "5", "G": "6", "B": "8", "P": "9",
}
DIGIT_TO_SERIES = {
    "0": "D", "1": "T", "2": "Z", "3": "E", "4": "A",
    "5": "S", "6": "G", "7": "T", "8": "B", "9": "P",
}
INVALID_SERIES_TO_VALID = {"I": "T", "J": "T", "O": "D", "Q": "D", "R": "B", "W": "V"}


@dataclass(frozen=True)
class PlateCandidate:
    text: str
    corrections: int
    series_length: int
    tail_length: int
    kind: str


def clean_plate_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def _digit_option(char: str):
    if char.isdigit():
        return char, 0
    mapped = CHAR_TO_DIGIT.get(char)
    return (mapped, 1) if mapped is not None else None


def _series_option(char: str):
    if char in SERIES_LETTERS:
        return char, 0
    if char == "Q":
        return char, 0
    if char in DIGIT_TO_SERIES:
        return DIGIT_TO_SERIES[char], 1
    mapped = INVALID_SERIES_TO_VALID.get(char)
    return (mapped, 1) if mapped is not None else None


def _candidate_from_pattern(raw: str, pattern: str, kind: str) -> PlateCandidate | None:
    output: list[str] = []
    corrections = 0
    for char, token in zip(raw, pattern):
        option = _series_option(char) if token == "S" else _digit_option(char)
        if option is None:
            return None
        normalized, cost = option
        output.append(normalized)
        corrections += cost

    candidate = "".join(output)
    if candidate[:2] not in VALID_PROVINCE_CODES:
        return None

    series_start = 2
    series_end = next(
        index for index in range(series_start, len(pattern))
        if pattern[index] == "N"
    )
    series = candidate[series_start:series_end]
    if kind == "car_special" and series not in SPECIAL_SERIES:
        return None

    return PlateCandidate(
        text=candidate,
        corrections=corrections,
        series_length=series_end - series_start,
        tail_length=len(candidate) - series_end,
        kind=kind,
    )


def plate_candidates(
    text: str,
    allow_motorbike_subseries: bool = True,
    allow_general_two_letter: bool = True,
) -> list[PlateCandidate]:
    """Generate position-aware candidates without inserting/deleting characters."""
    raw = clean_plate_text(text)
    if not 7 <= len(raw) <= 10:
        return []

    candidates: list[PlateCandidate] = []
    patterns = [
        ("NNS" + ("N" * 4), "single_series"),
        ("NNS" + ("N" * 5), "single_series"),
    ]
    if allow_motorbike_subseries:
        patterns.append(("NNS" + ("N" * 6), "single_series_long"))
    if allow_general_two_letter:
        patterns.extend([
            ("NNSS" + ("N" * 4), "two_letter"),
            ("NNSS" + ("N" * 5), "two_letter"),
        ])
    else:
        patterns.extend([
            ("NNSS" + ("N" * 4), "car_special"),
            ("NNSS" + ("N" * 5), "car_special"),
        ])
    if allow_motorbike_subseries:
        patterns.extend([
            ("NNSN" + ("N" * 4), "motorbike_subseries"),
            ("NNSN" + ("N" * 5), "motorbike_subseries"),
        ])

    for pattern, kind in patterns:
        if len(raw) != len(pattern):
            continue
        candidate = _candidate_from_pattern(raw, pattern, kind)
        if candidate is not None:
            candidates.append(candidate)

    kind_rank = {
        "single_series": 0,
        "single_series_long": 1,
        "two_letter": 1,
        "car_special": 1,
        "motorbike_subseries": 2,
    }
    candidates.sort(key=lambda item: (item.corrections, kind_rank.get(item.kind, 9), item.text))
    return candidates


def best_plate_candidate(
    text: str,
    preferred_series_length: int | None = None,
    allow_motorbike_subseries: bool = True,
    allow_general_two_letter: bool = True,
) -> PlateCandidate | None:
    candidates = plate_candidates(
        text,
        allow_motorbike_subseries=allow_motorbike_subseries,
        allow_general_two_letter=allow_general_two_letter,
    )
    if preferred_series_length is not None:
        preferred_candidates = [
            candidate for candidate in candidates
            if candidate.series_length == preferred_series_length
        ]
        if preferred_candidates:
            return preferred_candidates[0]
    return candidates[0] if candidates else None


def normalize_plate_text(text: str) -> str:
    candidate = best_plate_candidate(text)
    return candidate.text if candidate is not None else clean_plate_text(text)


def is_valid_plate(
    text: str,
    allow_motorbike_subseries: bool = True,
    allow_general_two_letter: bool = True,
) -> bool:
    raw = clean_plate_text(text)
    candidate = best_plate_candidate(
        raw,
        allow_motorbike_subseries=allow_motorbike_subseries,
        allow_general_two_letter=allow_general_two_letter,
    )
    return candidate is not None and candidate.text == raw
