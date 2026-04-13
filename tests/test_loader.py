from pathlib import Path


def test_llm01_manual_prompts_file_exists():
    assert Path("src/attacks/manual/llm01.txt").exists()


def test_llm01_manual_prompts_count():
    lines = [
        line.strip()
        for line in Path("src/attacks/manual/llm01.txt").read_text().splitlines()
        if line.strip()
    ]
    assert len(lines) == 5, f"Expected 5 prompts, got {len(lines)}"


def test_llm01_manual_prompts_non_empty():
    lines = [
        line.strip()
        for line in Path("src/attacks/manual/llm01.txt").read_text().splitlines()
        if line.strip()
    ]
    assert all(len(line) > 10 for line in lines), "Every prompt must be >10 characters"
