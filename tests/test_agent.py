import importlib
import pytest


@pytest.fixture()
def agent(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WORKSPACE", str(tmp_path))
    import agent as module
    return importlib.reload(module)


def test_parse_file_blocks(agent):
    text = "ок\n<<<FILE a/b.txt>>>\nпривет\n<<<END>>>\nещё\n<<<FILE c.md>>>\n```py\nx\n```\n<<<END>>>"
    assert agent.parse_file_blocks(text) == [("a/b.txt", "привет"), ("c.md", "```py\nx\n```")]


def test_write_file_creates_nested(agent):
    path = agent.write_file("site/index.html", "<h1>hi</h1>")
    assert path.read_text(encoding="utf-8") == "<h1>hi</h1>"


@pytest.mark.parametrize("bad", ["../evil.txt", "/etc/passwd", "a/../../x"])
def test_safe_path_blocks_escape(agent, bad):
    with pytest.raises(ValueError):
        agent.safe_path(bad)


def test_history_roundtrip_and_trim(agent, tmp_path):
    path = tmp_path / "h.json"
    msgs = [agent.system_message()] + [{"role": "user", "content": str(i)} for i in range(100)]
    agent.save_history(msgs, path)
    loaded = agent.load_history(path)
    assert loaded[0]["role"] == "system"
    assert len(loaded) == agent.MAX_MESSAGES + 1
    assert loaded[-1]["content"] == "99"


def test_corrupt_history_is_backed_up(agent, tmp_path):
    path = tmp_path / "h.json"
    path.write_text("{не json", encoding="utf-8")
    loaded = agent.load_history(path)
    assert loaded == [agent.system_message()]
    assert (tmp_path / "h.json.bad").exists()
