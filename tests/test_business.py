import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inputs import make_records
from business import segment_statistics, sku_prices


class BusinessTests(unittest.TestCase):
    def test_annotation_dedup_preserves_other_sku(self):
        df = pd.DataFrame({'评论内容':['好吃']*3, '评论ID':['a','a','b'], 'SKUID':['1','1','2'], '平台':['tmall']*3, 'SKU价格':['10','10','20'], '极性（1正0负2中）':['1','0','1']})
        rows = make_records(df,'评论内容',30)
        self.assertEqual(len(rows),2)
        self.assertNotIn('极性（1正0负2中）',rows[0])
        self.assertEqual(len(sku_prices(rows)),2)

    def test_make_records_uses_same_latest_comment_rule(self):
        df = pd.DataFrame({
            '月份':['2025-01-01', '2025-03-01'], '评论内容':['旧评价', '新评价'],
            '评论ID':['a', 'a'], 'SKUID':['1', '1'], '平台':['tmall', 'tmall']})
        rows = make_records(df, '评论内容', 30)
        self.assertEqual([row['text'] for row in rows], ['新评价'])

    def test_segment_denominator(self):
        reviews = [{'review_id':str(i),'platform':platform,'kind':'消费体验'} for i, platform in enumerate(['a','a','b'])]
        opinions = [{'review_id':str(i),'category':'口味','sentiment':'负面'} for i in [0,2]]
        rows = segment_statistics(reviews,opinions,'platform').set_index('分组')
        self.assertEqual(rows.loc['a','提及率'],.5)
        self.assertEqual(rows.loc['b','提及率'],1)
