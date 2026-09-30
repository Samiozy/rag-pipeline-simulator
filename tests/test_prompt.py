from rag.generation.extractive import ExtractiveGenerator
from rag.prompts import build_rag_prompt, build_retrieval_query


def test_prompt_contains_question_and_source():
    prompt = build_rag_prompt("What is RAG?", [("RAG combines retrieval and generation.", {"source": "sample.txt"})])
    assert "What is RAG?" in prompt
    assert "sample.txt" in prompt
    assert "RAG combines retrieval" in prompt
    assert "CONVERSATION" not in prompt


def test_prompt_includes_conversation_history():
    prompt = build_rag_prompt(
        "Why does that matter?",
        [("Retrieval quality can be judged from ranks and scores.", {"source": "sample.txt"})],
        history=[("Why should retrieval be evaluated independently from generation?", "Because a bad answer may come from retrieval, not the model.")],
    )
    assert "CONVERSATION" in prompt
    assert "Why should retrieval be evaluated independently from generation?" in prompt
    assert "Why does that matter?" in prompt
    assert "CONTEXT" in prompt


def test_retrieval_query_includes_prior_questions():
    query = build_retrieval_query(
        "What about overlap?",
        history=[("How does chunk size affect retrieval quality?", "Smaller chunks are more specific.")],
    )
    assert query.startswith("What about overlap?")
    assert "How does chunk size affect retrieval quality?" in query


def test_retrieval_query_without_history_is_unchanged():
    assert build_retrieval_query("What is RAG?") == "What is RAG?"


def test_extractive_generator_reads_question_after_conversation():
    prompt = build_rag_prompt(
        "What is chunk overlap used for?",
        [(
            "Chunk overlap is commonly used to reduce information loss at chunk boundaries. "
            "Very small chunks can improve specificity but may lose surrounding context.",
            {"source": "sample.txt"},
        )],
        history=[("How does chunk size affect retrieval quality?", "Smaller chunks are more specific.")],
    )
    answer = ExtractiveGenerator().generate(prompt)
    assert "overlap" in answer.lower()
