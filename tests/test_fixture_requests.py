from scripts.generate_fixture_requests import PINYIN


def test_fixture_request_count_and_uniqueness():
    assert len(PINYIN) == 100
    assert len(set(PINYIN)) == 100
