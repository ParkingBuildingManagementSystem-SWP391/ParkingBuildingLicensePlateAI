import unittest

import numpy as np
import torch

from app.recognition.crnn import ALPHABET, CRNN, greedy_decode, normalize_plate_image
from app.recognition.plate_rules import is_valid_plate, normalize_plate_text
from app.utils.helpers import correct_license_plate_vietnam, merge_close_boxes
from training.prepare_ocr_dataset import split_groups, split_two_line_crop
from training.import_yolo_ocr_dataset import is_plausible_full_plate


class RecognitionTests(unittest.TestCase):
    def test_model_output_shape(self):
        model = CRNN()
        output = model(torch.zeros(2, 1, 32, 160))
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[2], len(ALPHABET) + 1)

    def test_normalization_shape(self):
        image = np.full((60, 180, 3), 255, dtype=np.uint8)
        tensor = normalize_plate_image(image)
        self.assertEqual(tuple(tensor.shape), (1, 32, 160))
        self.assertGreaterEqual(float(tensor.min()), -1.0)
        self.assertLessEqual(float(tensor.max()), 1.0)

    def test_ctc_greedy_decode(self):
        # Blank, 5, repeated 5, blank, A => "5A"
        token_ids = [0, ALPHABET.index("5") + 1, ALPHABET.index("5") + 1, 0, ALPHABET.index("A") + 1]
        logits = torch.full((1, len(token_ids), len(ALPHABET) + 1), -10.0)
        for step, token_id in enumerate(token_ids):
            logits[0, step, token_id] = 10.0
        self.assertEqual(greedy_decode(logits), ["5A"])

    def test_two_line_split_uses_plate_structure(self):
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        samples = split_two_line_crop(image, "59X312345")
        self.assertEqual([label for _, label in samples], ["59X3", "12345"])

    def test_group_split_has_no_plate_leakage(self):
        labels = [f"59A{i:05d}" for i in range(20)]
        assignment = split_groups(labels, seed=42, train_ratio=0.8, val_ratio=0.1)
        self.assertEqual(len(assignment), len(set(labels)))
        self.assertEqual(set(assignment.values()), {"train", "val", "test"})

    def test_external_filename_must_contain_full_plate(self):
        self.assertTrue(is_plausible_full_plate("51F86947"))
        self.assertTrue(is_plausible_full_plate("59X312345"))
        self.assertFalse(is_plausible_full_plate("17054"))
        self.assertFalse(is_plausible_full_plate("NULL"))

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
