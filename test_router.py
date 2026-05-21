"""
Smoke test for the LLM router.
Usage: python test_router.py
"""

import time
from llm_router import call_llm, cache_stats, clear_cache


def test_gemini():
    print("Test 1: Gemini call (fundamentals_analyst role)")
    response = call_llm(
        prompt="In one sentence: what does P/E ratio measure?",
        role="fundamentals_analyst",
        use_cache=False,
    )
    print(f"  Response: {response.strip()[:200]}")
    assert len(response) > 10, "Empty response from Gemini"
    print("  PASS\n")


def test_groq():
    print("Test 2: Groq call (bull_researcher role)")
    response = call_llm(
        prompt="In one sentence: what does it mean to be bullish on a stock?",
        role="bull_researcher",
        use_cache=False,
    )
    print(f"  Response: {response.strip()[:200]}")
    assert len(response) > 10, "Empty response from Groq"
    print("  PASS\n")


def test_caching():
    print("Test 3: Caching")
    # Clear cache first to guarantee a clean slate
    clear_cache()

    prompt = "What is 2+2? Reply with just the number."

    start = time.time()
    response1 = call_llm(prompt=prompt, role="fundamentals_analyst")
    first_duration = time.time() - start
    print(f"  First call:  {first_duration:.3f}s  -> {response1.strip()[:50]}")

    start = time.time()
    response2 = call_llm(prompt=prompt, role="fundamentals_analyst")
    second_duration = time.time() - start
    print(f"  Second call: {second_duration:.3f}s -> {response2.strip()[:50]}")

    assert response1 == response2, "Cache returned different response!"
    assert first_duration > 0.1, f"First call too fast ({first_duration:.3f}s) — was cache really empty?"
    assert second_duration < 0.1, f"Second call too slow ({second_duration:.3f}s) — cache not working"
    print("  PASS\n")


def test_invalid_role():
    print("Test 4: Invalid role raises clear error")
    try:
        call_llm(prompt="hello", role="nonexistent_role")
        print("  FAIL — should have raised ValueError")
        return False
    except ValueError as e:
        print(f"  Got expected error: {str(e)[:80]}...")
        print("  PASS\n")
        return True


def main():
    print("=" * 60)
    print("LLM Router Smoke Test")
    print("=" * 60)
    print()

    test_gemini()
    test_groq()
    test_caching()
    test_invalid_role()

    print("=" * 60)
    print("Cache stats:", cache_stats())
    print("=" * 60)
    print("\nAll tests passed. Router is working.")


if __name__ == "__main__":
    main()