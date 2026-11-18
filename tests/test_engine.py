import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import Model, validate, statistics, discover, opportunities


class FakeModel:
    def __init__(self, result):
        self.result = result

    def ask(self, *args):
        return self.result


class SequenceModel:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = 0

    def ask(self, *args):
        self.calls += 1
        return next(self.results)


class EngineTests(unittest.TestCase):
    def opinion(self, **extra):
        return dict(aspect='甜度', opinion='太甜', sentiment='负面', evidence='太甜', scene='', explicit_need='', inferred_need='', **extra)

    def test_evidence_must_be_original(self):
        with self.assertRaises(ValueError):
            validate({'kind': '消费体验', 'opinions': [self.opinion()]}, '很好吃')
        self.assertEqual(len(validate({'kind': '消费体验', 'opinions': [self.opinion()]}, '味道太甜')['opinions']), 1)

    def test_non_experience_cannot_have_opinions(self):
        with self.assertRaises(ValueError):
            validate({'kind': '广告文案', 'opinions': [self.opinion()]}, '太甜')

    def test_counts_unique_reviews(self):
        rows = [self.opinion(review_id='1', opinion_id=i, category='甜度') for i in range(2)]
        rows.append({**self.opinion(review_id='2', opinion_id=2, category='甜度'), 'sentiment': '正面'})
        stats = statistics(rows, 4).iloc[0]
        self.assertEqual(stats['提及评论数'], 2)
        self.assertEqual(stats['提及率'], .5)
        self.assertEqual(stats['含负面率'], .5)

    def test_incomplete_assignments_rejected(self):
        with self.assertRaises(ValueError):
            discover(FakeModel({'categories': ['甜度'], 'assignments': []}), [self.opinion(opinion_id=0)])

    def test_incomplete_assignments_are_repaired_once(self):
        model = SequenceModel([
            {'categories': ['甜度'], 'assignments': []},
            {'categories': ['甜度'], 'assignments': [{'opinion_id': 0, 'category': '甜度'}]},
        ])
        rows = discover(model, [self.opinion(opinion_id=0)])
        self.assertEqual(rows[0]['category'], '甜度')
        self.assertEqual(model.calls, 2)

    def test_invalid_report_reference_rejected(self):
        rows = [self.opinion(review_id='1', opinion_id=0, category='甜度')]
        item = dict(direction='低糖', basis='推断', validation='测试', tradeoff='口感', evidence_ids=[999])
        with self.assertRaises(ValueError):
            opportunities(FakeModel({'opportunities': [item]}), rows, statistics(rows, 1))

    def test_opportunity_count_limit_is_enforced(self):
        rows = [self.opinion(review_id='1', opinion_id=0, category='甜度')]
        item = dict(direction='低糖', basis='依据', validation='测试', tradeoff='口感', evidence_ids=[0])
        with self.assertRaisesRegex(ValueError, '超过8条'):
            opportunities(FakeModel({'opportunities': [item] * 9}), rows, statistics(rows, 1))

    @patch('engine.requests.post')
    def test_gpt_56_sol_uses_responses_api(self, post):
        response = Mock(ok=True, status_code=200)
        response.json.return_value = {
            'id': 'resp_test',
            'status': 'completed',
            'service_tier': 'default',
            'usage': {'input_tokens': 12, 'output_tokens': 4},
            'output': [{'type': 'message', 'content': [
                {'type': 'output_text', 'text': '{"ok":true}'}
            ]}],
        }
        post.return_value = response
        client = Model('sk-test', 'https://api.openai.com/v1', 'gpt-5.6-sol', reasoning_effort='low')
        self.assertEqual(client.ask('只返回 JSON。', {'task': 'test'}), {'ok': True})
        url = post.call_args.args[0]
        payload = post.call_args.kwargs['json']
        self.assertEqual(url, 'https://api.openai.com/v1/responses')
        self.assertEqual(payload['model'], 'gpt-5.6-sol')
        self.assertEqual(payload['text']['format'], {'type': 'json_object'})
        self.assertEqual(payload['reasoning'], {'effort': 'low'})
        self.assertFalse(payload['store'])
        self.assertEqual(client.last_metadata['response_id'], 'resp_test')

    @patch('engine.requests.post')
    def test_chat_completions_compatibility_mode(self, post):
        response = Mock(ok=True, status_code=200)
        response.json.return_value = {
            'id': 'chat_test',
            'choices': [{'message': {'content': '{"ok":true}'}}],
        }
        post.return_value = response
        client = Model('key', 'https://example.test/v1', 'compatible-model', api_mode='chat_completions')
        self.assertEqual(client.ask('返回 JSON。', {}), {'ok': True})
        self.assertEqual(post.call_args.args[0], 'https://example.test/v1/chat/completions')


if __name__ == '__main__':
    unittest.main()
