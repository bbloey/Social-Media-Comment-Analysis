import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard import prepare_data, filter_data, token_frequencies, price_bands, unique_sku_observations


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.mapping = {'month':'月份','platform':'平台','brand':'品牌','price':'SKU价格',
            'product_price':'价格','sku_id':'SKUID','sku_name':'SKU名称','comment_id':'评论ID'}
        self.frame = pd.DataFrame({
            '月份':['2025-01-01','2025-03-01','2025-02-01','2025-02-01'],
            '平台':['a','a','b','b'], '品牌':['甲','甲','乙','乙'], 'SKU价格':['10','12','25','bad'],
            '价格':['9','11','23','30'], 'SKUID':['s1','s1','s2','s2'], 'SKU名称':['x','x','y','y'],
            '评论ID':['c1','c1','c2','c3'], '评论内容':['旧评价','新评价','巧克力味道很好','   ']})

    def test_comment_id_keeps_latest(self):
        data, quality = prepare_data(self.frame, '评论内容', self.mapping, '评论ID优先')
        self.assertEqual(len(data), 2)
        self.assertNotIn('旧评价', data['评论内容'].tolist())
        self.assertEqual(quality['去除重复行数'], 1)
        self.assertEqual(quality['价格可解析数'], 2)

    def test_filters(self):
        data, _ = prepare_data(self.frame, '评论内容', self.mapping)
        result = filter_data(data, self.mapping, platforms=['b'], price_range=(20,30))
        self.assertEqual(result['评论ID'].tolist(), ['c2'])

    def test_tokenization_and_price_band(self):
        frequencies = token_frequencies(['巧克力味道很好，巧克力很香'], {'味道'})
        self.assertEqual(frequencies['巧克力'], 2)
        self.assertEqual(str(price_bands(pd.Series([9,15,25,60,150])).iloc[1]), '10–20')

    def test_price_analysis_is_sku_level(self):
        data, _ = prepare_data(self.frame, '评论内容', self.mapping)
        sku = unique_sku_observations(data, self.mapping)
        self.assertEqual(sku['SKUID'].nunique(), len(sku))


if __name__ == '__main__':
    unittest.main()
