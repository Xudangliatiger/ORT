import json
import pytest
from scripts.prepare_dataset import records, validate_record


def test_conversion_preserves_labels_and_all_crops(tmp_path):
    crops = [list(range(256)), list(reversed(range(256)))]
    source = tmp_path / 'tokens.jsonl'
    source.write_text(json.dumps({'class_id': 17, 'codes': crops}) + '\n')
    assert list(records([str(source)], 'maskgit', 'class_id', 'codes')) == [{'label': 17, 'tokens': crops}]
    assert validate_record({'label': 17, 'tokens': crops[0]}, 'maskgit')['tokens'] == [crops[0]]


@pytest.mark.parametrize('label,tokens', [(1000, [0]*256), (0, [0]*273), (0, [1024]*256), (0, [1.0]*256), (0, [])])
def test_invalid_maskgit_records_fail(label, tokens):
    with pytest.raises(ValueError):
        validate_record({'label': label, 'tokens': tokens}, 'maskgit')
