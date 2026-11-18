import json
import re
import sys
from pathlib import Path


def main():
    app_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(app_root))
    try:
        from features import changelog_catalog, knowledge_base_info, load_rules, load_taxonomy
        taxonomy = load_taxonomy()
        rules = load_rules()
        example_errors = []
        for rule in rules:
            examples = [item.strip() for item in rule.get('examples', '').split('|') if item.strip()]
            unmatched = [example for example in examples if re.search(rule['pattern'], example, flags=re.I) is None]
            if unmatched:
                example_errors.append({'rule_id': rule['rule_id'], 'unmatched_examples': unmatched})
        if example_errors:
            raise ValueError(f'Rule examples failed to match: {example_errors}')
        if '品类' not in {row['feature'] for row in taxonomy}:
            raise ValueError('Top-level taxonomy must contain 品类 for business data labels')
        info = knowledge_base_info()
        changelog = changelog_catalog()
        if not changelog or info['knowledge_base_version'] not in {row.get('kb_version') for row in changelog}:
            raise ValueError('Knowledge-base changelog must contain the current knowledge_base_version')
        print(json.dumps({'ok': True, **info, 'changelog_entries': len(changelog),
            'example_checks': sum(len([x for x in row.get('examples', '').split('|') if x.strip()]) for row in rules)},
            ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({'ok': False, 'error': f'{type(exc).__name__}: {exc}'}, ensure_ascii=False, indent=2))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
