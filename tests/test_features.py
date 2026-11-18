import sys
import unittest
import csv
import os
import shutil
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features import (DEFAULT_LEXICON_PATH, DEFAULT_TAXONOMY_PATH, extract_feature_rows,
    extract_text_features, knowledge_base_info, taxonomy_catalog)


class FeatureTests(unittest.TestCase):
    def row_for(self, text, rule_id):
        return next(row for row in extract_text_features(text) if row['rule_id'] == rule_id)

    def test_user_example_is_multi_label(self):
        text = '价格太贵了而且一袋太多了吃不完'
        rows = extract_text_features(text)
        by_feature = {row['feature']: row for row in rows}
        self.assertEqual(by_feature['价格与价值']['standard_tag'], '价格偏高')
        self.assertEqual(by_feature['包装与规格']['standard_tag'], '单包份量过大')
        self.assertEqual(by_feature['包装与规格']['evidence'], '一袋太多了吃不完')
        for row in rows:
            self.assertEqual(text[row['start']:row['end']], row['evidence'])

    def test_taste_and_texture(self):
        rows = extract_text_features('甜味适中，口感酥脆，也有点咸')
        tags = {row['standard_tag'] for row in rows}
        self.assertTrue({'甜味', '咸味', '酥脆'}.issubset(tags))

    def test_value_for_money_is_positive(self):
        row = self.row_for('性价比高', 'PRICE_LOW')
        self.assertEqual(row['polarity_candidate'], '正面')
        self.assertEqual(row['statement_type'], '体验')

    def test_delivery_speed_has_directional_polarity(self):
        self.assertEqual(self.row_for('物流很快', 'SERVICE_DELIVERY')['polarity_candidate'], '正面')
        self.assertEqual(self.row_for('物流很慢', 'SERVICE_DELIVERY')['polarity_candidate'], '负面')

    def test_negated_negative_is_not_negative(self):
        row = self.row_for('不是很贵', 'PRICE_HIGH')
        self.assertEqual(row['polarity_candidate'], '中性')
        self.assertTrue(row['negated'])

    def test_not_too_sweet_is_neutral(self):
        row = self.row_for('不太甜', 'TASTE_SWEET')
        self.assertEqual(row['polarity_candidate'], '中性')

    def test_request_is_not_consumption_sentiment(self):
        row = self.row_for('希望甜一点', 'TASTE_SWEET')
        self.assertEqual(row['statement_type'], '诉求')
        self.assertEqual(row['polarity_candidate'], '不适用')

    def test_contrast_keeps_each_aspect_in_its_segment(self):
        rows = extract_text_features('价格有点贵，但是味道很好吃')
        by_rule = {row['rule_id']: row for row in rows}
        self.assertEqual(by_rule['PRICE_HIGH']['polarity_candidate'], '负面')
        self.assertEqual(by_rule['TASTE_OVERALL']['polarity_candidate'], '正面')
        self.assertNotEqual(by_rule['PRICE_HIGH']['clause'], by_rule['TASTE_OVERALL']['clause'])

    def test_business_category_is_data_label(self):
        rows = extract_feature_rows([{'review_id':'1', 'text':'好吃', 'category':'夹心饼干'}])
        category = [row for row in rows if row['feature'] == '品类'][0]
        self.assertEqual(category['standard_tag'], '夹心饼干')
        self.assertEqual(category['source'], '数据标签')
        self.assertEqual(category['polarity_candidate'], '不适用')

    def test_top_level_taxonomy_is_bounded(self):
        taxonomy = taxonomy_catalog()
        self.assertGreater(len(taxonomy), 4)
        self.assertLessEqual(len(taxonomy), 10)
        self.assertEqual(len({row['feature'] for row in taxonomy}), len(taxonomy))

    def test_external_lexicon_can_be_extended_without_code_change(self):
        old_lexicon = os.environ.get('REVIEW_FEATURE_LEXICON')
        old_taxonomy = os.environ.get('REVIEW_FEATURE_TAXONOMY')
        with tempfile.TemporaryDirectory() as folder:
            lexicon = Path(folder) / 'feature_lexicon.csv'
            taxonomy = Path(folder) / 'feature_taxonomy.csv'
            shutil.copyfile(DEFAULT_LEXICON_PATH, lexicon)
            shutil.copyfile(DEFAULT_TAXONOMY_PATH, taxonomy)
            with DEFAULT_LEXICON_PATH.open(encoding='utf-8-sig', newline='') as source:
                fieldnames = csv.DictReader(source).fieldnames
            os.environ['REVIEW_FEATURE_LEXICON'] = str(lexicon)
            os.environ['REVIEW_FEATURE_TAXONOMY'] = str(taxonomy)
            try:
                initial_version = knowledge_base_info()['knowledge_base_version']
                with lexicon.open('a', encoding='utf-8', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writerow({'enabled':1, 'priority':20, 'rule_id':'TASTE_MILK_WEAK', 'feature':'口味',
                        'standard_tag':'奶味不足', 'pattern':'奶味不足', 'default_polarity':'负面',
                        'statement_type':'体验', 'polarity_strategy':'fixed', 'positive_context':'',
                        'negative_context':'', 'neutral_context':'', 'basis':'奶味强度不足',
                        'examples':'奶味不足', 'version_note':'测试扩展', 'notes':''})
                rows = extract_text_features('这个饼干奶味不足')
                self.assertTrue(any(row['standard_tag'] == '奶味不足' for row in rows))
                self.assertNotEqual(knowledge_base_info()['knowledge_base_version'], initial_version)
            finally:
                if old_lexicon is None:
                    os.environ.pop('REVIEW_FEATURE_LEXICON', None)
                else:
                    os.environ['REVIEW_FEATURE_LEXICON'] = old_lexicon
                if old_taxonomy is None:
                    os.environ.pop('REVIEW_FEATURE_TAXONOMY', None)
                else:
                    os.environ['REVIEW_FEATURE_TAXONOMY'] = old_taxonomy


if __name__ == '__main__':
    unittest.main()
