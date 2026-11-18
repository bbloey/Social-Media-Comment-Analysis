import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit import audit_text, audit_csv_rows
from engine import extract


class AuditTests(unittest.TestCase):
    def test_csv_keeps_hits_and_uncovered_text(self):
        import csv
        import io
        import pandas as pd
        texts = ['太甜，\"不好吃\"\n不好撕', '今天出门']
        audits = [{'review_id': str(i), 'text': text, **audit_text(text)} for i, text in enumerate(texts)]
        rows = audit_csv_rows(audits)
        encoded = pd.DataFrame(rows).to_csv(index=False).encode('utf-8-sig')
        decoded = list(csv.DictReader(io.StringIO(encoded.decode('utf-8-sig'))))
        self.assertEqual(len(decoded), len(audits[0]['traces']) + 1)
        self.assertEqual(decoded[0]['text'], texts[0])
        self.assertEqual(decoded[-1]['text'], texts[1])
        self.assertEqual(decoded[-1]['rule_id'], '')
        self.assertIn('规则未覆盖', decoded[-1]['decision'])

    def test_offsets(self):
        text = '味道好吃，但是太甜，封口不好撕'
        traces = audit_text(text)['traces']
        self.assertGreaterEqual(len(traces), 3)
        for row in traces:
            self.assertEqual(text[row['start']:row['end']], row['evidence'])

    def test_negation(self):
        rows = audit_text('不太甜，不好吃')['traces']
        self.assertTrue(any(r['rule_id'] == 'TASTE_SWEET' and r['sentiment_candidate'] == '中性' for r in rows))
        self.assertTrue(any(r['sentiment_candidate'] == '负面' and r['evidence'] == '不好吃' for r in rows))

    def test_audit_uses_same_rule_and_version_as_feature_engine(self):
        rows = audit_text('性价比高')['traces']
        self.assertEqual(rows[0]['rule_id'], 'PRICE_LOW')
        self.assertEqual(rows[0]['sentiment_candidate'], '正面')
        self.assertIn(rows[0]['knowledge_base_version'], rows[0]['rule_version'])

    def test_uncovered(self):
        self.assertEqual(audit_text('今天出门')['traces'], [])

    def test_failed_api_keeps_audit(self):
        class Broken:
            base, model = 'offline', 'test'
            def ask(self, *args):
                raise RuntimeError('模拟接口失败')
        with tempfile.TemporaryDirectory() as root:
            reviews, opinions, errors = extract(Broken(), [{'review_id': '1', 'text': '太甜'}], root)
        self.assertFalse(reviews)
        self.assertFalse(opinions)
        self.assertEqual(errors[0]['rule_audit']['traces'][0]['evidence'], '太甜')
