from rag.prompts import build_rag_prompt


def test_prompt_contains_question_and_source():
    prompt = build_rag_prompt("What is RAG?", [("RAG combines retrieval and generation.", {"source": "sample.txt"})])
    assert "What is RAG?" in prompt
    assert "sample.txt" in prompt
    assert "RAG combines retrieval" in prompt
