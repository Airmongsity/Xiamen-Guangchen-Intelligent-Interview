"""Query-aware history gate: retrospective-language detector (regex, no API)."""

import pytest

from automem.retrieval import is_retrospective


@pytest.mark.parametrize("q", [
    "Where did the user used to work?",
    "What was her job previously?",
    "Where did they live before moving?",
    "What phone did I have formerly?",
    "Which city was she in earlier?",
    "他以前在哪里工作?",
    "她之前养的是什么宠物?",
    "原来的公司是哪家?",
    "他曾经住在哪个城市?",
])
def test_retrospective_true(q):
    assert is_retrospective(q)


@pytest.mark.parametrize("q", [
    "Where does the user work?",
    "What is her current job?",
    "Which city does she live in now?",
    "What pet does he have?",
    "他现在在哪里工作?",
])
def test_retrospective_false(q):
    assert not is_retrospective(q)
