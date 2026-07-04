import pytest

from mini_agent.tools.calculator import calculate


def test_basic_arithmetic():
    assert calculate(None, expression="2 * (3 + 4)")["result"] == 14
    assert calculate(None, expression="2 ** 10")["result"] == 1024
    assert calculate(None, expression="7 // 2")["result"] == 3
    assert calculate(None, expression="-5 + 3")["result"] == -2


def test_rejects_names_and_calls():
    # Anything that is not pure arithmetic must raise, not execute.
    for expr in ["__import__('os').system('echo hi')", "a + 1", "len([1,2])"]:
        with pytest.raises((ValueError, SyntaxError)):
            calculate(None, expression=expr)
