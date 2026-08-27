from data_pipeline.build_examples import split_for


def test_document_split_is_stable_and_exclusive():
    first = split_for("zhwiki:42")
    assert first in {"train", "val", "test"}
    assert split_for("zhwiki:42") == first


def test_document_split_has_all_partitions():
    observed = {split_for(f"fixture:{index}") for index in range(500)}
    assert observed == {"train", "val", "test"}
