import unittest

import cv2
import numpy as np

from app.recognition.plate_rules import is_valid_plate, normalize_plate_text
from app.utils.plate_image import preprocess_plate_variants, TARGET_CROP_WIDTH
from app.utils.helpers import (
    correct_license_plate_vietnam,
    merge_close_boxes,
    ocr_result_confidence,
    select_best_ocr_candidate,
)


class RecognitionTests(unittest.TestCase):
    def test_plate_preprocessing_returns_contrast_and_binary_variants(self):
        image = np.full((60, 180, 3), 220, dtype=np.uint8)
        cv2.putText(image, "59A12345", (5, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)

        variants = preprocess_plate_variants(image)

        self.assertEqual(variants.contrasted.shape, variants.binary.shape)
        self.assertEqual(variants.binary.shape[1], TARGET_CROP_WIDTH)
        self.assertTrue(set(np.unique(variants.binary)).issubset({0, 255}))

    def test_valid_plate_wins_over_higher_confidence_invalid_text(self):
        candidates = [
            ("contrast", "59X312345", 0.72),
            ("binary", "5931234", 0.95),
        ]
        self.assertEqual(select_best_ocr_candidate(candidates)[0], "contrast")

    def test_ocr_confidence_is_weighted_by_character_count(self):
        results = [
            ([[0, 0]] * 4, "59A", 0.9),
            ([[0, 0]] * 4, "12345", 0.6),
        ]
        self.assertAlmostEqual(ocr_result_confidence(results), (3 * 0.9 + 5 * 0.6) / 8)

    def test_position_aware_character_correction(self):
        self.assertEqual(normalize_plate_text("3OA12I45"), "30A12145")
        self.assertEqual(normalize_plate_text("59X3I2345"), "59X312345")
        self.assertTrue(is_valid_plate("59X312345"))

    def test_valid_l0_and_m0_are_not_changed_to_special_series(self):
        self.assertEqual(correct_license_plate_vietnam("50M09666"), "50M09666")
        self.assertEqual(correct_license_plate_vietnam("28L08349"), "28L08349")
        self.assertEqual(correct_license_plate_vietnam("80LD12345"), "80LD12345")
        self.assertEqual(correct_license_plate_vietnam("80QT12345"), "80QT12345")

    def test_close_numeric_boxes_are_not_merged_into_a_letter(self):
        boxes = [
            {"text": "1", "center_x": 17.5, "min_x": 10, "max_x": 25, "width": 15},
            {"text": "8", "center_x": 38.5, "min_x": 30, "max_x": 47, "width": 17},
        ]
        self.assertEqual([box["text"] for box in merge_close_boxes(boxes)], ["1", "8"])


if __name__ == "__main__":
    unittest.main()
