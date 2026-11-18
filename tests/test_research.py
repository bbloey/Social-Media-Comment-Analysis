import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import analyze_signals, hierarchy, synthesize, validate_signals, validate_report, html_report, save_checkpoint
from inputs import TEST_FILE, read_source, make_records
from engine import statistics


class Model:
    base = 'test://offline'
    model = 'fixture'

    def ask(self, prompt, data):
        if 'existing' in data:
            raise AssertionError()
        if 'categories' in data:
            return {'tags': [{'level1': '产品功能', 'level2': '配方', 'category': c} for c in data['categories']]}
        if 'text' in data:
            return {'relevance': '无法判断', 'reason': '离线夹具不进行语义判断', 'signals': []}
        return {'summary': '离线流程测试，无市场结论', 'findings': [], 'gaps': ['需真实模型及人工复核']}


class ResearchTests(unittest.TestCase):
    def test_invalid_evidence_rejected(self):
        with self.assertRaises(ValueError):
            validate_signals({'relevance': '直接相关', 'reason': 'test', 'signals': [{'type': '痛点', 'claim': '太甜', 'evidence': '太甜'}]}, '好吃')

    def test_report_refs_rejected(self):
        with self.assertRaises(ValueError):
            validate_report({'summary': 'x', 'gaps': [], 'findings': [{'kind': '痛点', 'title': 'x', 'interpretation': 'x', 'next_test': 'x', 'tradeoff': 'x', 'evidence_ids': [8]}]}, [])

    def test_html_escapes_untrusted_text(self):
        report = {'summary': '<script>alert(1)</script>', 'coverage': {}, 'findings': [], 'evidence': [], 'gaps': []}
        rendered = html_report('<img src=x onerror=alert(1)>', report, statistics([], 0), [])
        self.assertNotIn('<script>alert(1)', rendered)
        self.assertNotIn('<img src=x', rendered)
        self.assertIn('&lt;script&gt;', rendered)

    def test_hierarchy_preserves_categories(self):
        result = hierarchy(Model(), [{'category': '甜度过高'}])
        self.assertEqual(result[0]['category'], '甜度过高')

    @unittest.skipUnless(TEST_FILE.exists(), '用户本地测试文件不存在')
    def test_actual_excel_offline_workflow_and_cache(self):
        records = make_records(read_source(TEST_FILE), 'original_text', 30)
        self.assertEqual(len(records), 22)
        with tempfile.TemporaryDirectory() as folder:
            rows, errors = analyze_signals(Model(), records, '零食开品', folder)
            self.assertFalse(errors)
            self.assertEqual(len(rows), 22)
            self.assertTrue(all(r['text'] == r['original_text'] for r in rows))
            report = synthesize(Model(), '零食开品', rows, [], statistics([], 0))
            self.assertEqual(report['coverage']['relevant_reviews'], 0)
            self.assertEqual(report['findings'], [])
            path = save_checkpoint(folder, 'test', 'report', report)
            self.assertEqual(json.loads((path / 'report.json').read_text(encoding='utf-8')), report)


if __name__ == '__main__':
    unittest.main()
