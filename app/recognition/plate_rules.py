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


def plate_candidates(text: str) -> list[PlateCandidate]:
    """Generate position-aware candidates without inserting/deleting characters."""
    raw = clean_plate_text(text)
    if not 7 <= len(raw) <= 10:
        return []

    candidates: list[PlateCandidate] = []
    for series_length in (1, 2):
        tail_length = len(raw) - 2 - series_length
        if tail_length not in (4, 5, 6):
            continue

        output: list[str] = []
        corrections = 0
        for index, char in enumerate(raw):
            option = _series_option(char) if 2 <= index < 2 + series_length else _digit_option(char)
            if option is None:
                break
            normalized, cost = option
            output.append(normalized)
            corrections += cost
        else:
            candidate = "".join(output)
            series = candidate[2:2 + series_length]
            series_is_allowed = (
                series in SPECIAL_SERIES
                if any(char not in SERIES_LETTERS for char in series)
                else True
            )
            if candidate[:2] in VALID_PROVINCE_CODES and series_is_allowed:
                candidates.append(PlateCandidate(candidate, corrections, series_length, tail_length))

    candidates.sort(key=lambda item: (item.corrections, item.series_length, item.text))
    return candidates


def best_plate_candidate(text: str, preferred_series_length: int | None = None) -> PlateCandidate | None:
    candidates = plate_candidates(text)
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


def is_valid_plate(text: str) -> bool:
    raw = clean_plate_text(text)
    candidate = best_plate_candidate(raw)
    return candidate is not None and candidate.text == raw
