import unittest

from src.inspection.defect_training.qwen_grounding import parse_qwen_grounding_boxes


class QwenGroundingParserTests(unittest.TestCase):
    def test_parses_fenced_normalized_json(self):
        response = '```json\n[{"bbox_2d":[100,200,400,600],"label":"vial"}]\n```'
        boxes, raw_count = parse_qwen_grounding_boxes(response, (2000, 1000), 0.7)
        self.assertEqual(raw_count, 1)
        self.assertEqual(boxes[0].as_list(), [200.0, 200.0, 800.0, 600.0])

    def test_clamps_and_skips_invalid_entries(self):
        response = '{"objects":[{"bbox_2d":[-10,900,1010,100]}, {"bbox_2d":[1,2,3]}]}'
        boxes, raw_count = parse_qwen_grounding_boxes(response, (1000, 500), 0.7)
        self.assertEqual(raw_count, 1)
        self.assertEqual(boxes[0].as_list(), [0.0, 50.0, 1000.0, 450.0])

    def test_invalid_response_is_empty(self):
        self.assertEqual(parse_qwen_grounding_boxes("not json", (100, 100), 0.7), ([], 0))

    def test_salvages_complete_boxes_from_truncated_array(self):
        response = '[{"bbox_2d":[100,200,300,400]}, {"bbox_2d":[500,600'
        boxes, raw_count = parse_qwen_grounding_boxes(response, (1000, 1000), 0.7)
        self.assertEqual(raw_count, 1)
        self.assertEqual(boxes[0].as_list(), [100.0, 200.0, 300.0, 400.0])


if __name__ == "__main__":
    unittest.main()
