import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import build_review_samples, evaluate_gold_standard, quick_review_metrics


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {'review_id': '1', 'text': '太甜'},
            {'review_id': '2', 'text': '没有命中'},
        ]
        self.features = [{
            'review_id': '1', 'source': '文本规则', 'rule_id': 'TASTE_SWEET', 'feature': '口味',
            'standard_tag': '甜味', 'evidence': '太甜', 'start': 0, 'end': 2,
            'polarity_candidate': '负面', 'statement_type': '体验',
        }]

    def test_sampling_separates_hit_and_uncovered(self):
        sample = build_review_samples(self.records, self.features, 10, 10)
        self.assertEqual(sample['hit_review_ids'], ['1'])
        self.assertEqual(sample['uncovered_review_ids'], ['2'])

    def test_quick_review_does_not_claim_recall(self):
        metrics = quick_review_metrics(
            [{'review_result': '正确'}], [{'coverage_review': '有遗漏'}])
        self.assertEqual(metrics['estimated_precision'], 1)
        self.assertEqual(metrics['uncovered_demand_discovery_rate'], 1)
        self.assertFalse(metrics['is_formal_recall'])

    def test_complete_gold_annotations_produce_recall(self):
        gold = [
            {'review_id':'1', 'text':'太甜', 'annotation_complete':True, 'gold_no_feature':False,
             'gold_feature':'口味', 'gold_standard_tag':'甜味', 'gold_evidence':'太甜',
             'gold_start':0, 'gold_end':2, 'gold_polarity':'负面', 'gold_statement_type':'体验'},
            {'review_id':'2', 'text':'没有命中', 'annotation_complete':True, 'gold_no_feature':True},
        ]
        metrics = evaluate_gold_standard(self.features, gold, ['1', '2'])
        self.assertEqual(metrics['precision_relaxed'], 1)
        self.assertEqual(metrics['recall_relaxed'], 1)
        self.assertEqual(metrics['polarity_accuracy_given_matched_feature'], 1)

    def test_incomplete_gold_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '尚未完成完整标注'):
            evaluate_gold_standard(self.features, [
                {'review_id':'1', 'text':'太甜', 'annotation_complete':False}
            ], ['1'])


if __name__ == '__main__':
    unittest.main()
