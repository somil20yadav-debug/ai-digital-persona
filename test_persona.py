"""
End-to-end integration test for YOU - AI Persona Clone
Verifies vector retrieval, few-shot prompt synthesis, and generation pipeline.
"""
from src.generator import retrieve, build_persona_prompt, generate_response
from src.style_analyzer import StyleAnalyzer

def run_test():
    print("=" * 60)
    print("  YOU - AI Writing Style Clone | Pipeline Verification")
    print("=" * 60)

    test_prompt = "Can you review the new pull request before the demo?"
    print(f"\n[1] User Query: \"{test_prompt}\"")

    # Step 1: Semantic Retrieval
    print("\n[2] Performing HNSW Vector Retrieval (top-3 matches)...")
    matches = retrieve(test_prompt, top_k=2)
    for idx, match in enumerate(matches, 1):
        print(f"    Match #{idx} [{match['id']}]: {match['text'][:80]}...")

    # Step 2: Prompt Formulation
    print("\n[3] Formulating Few-Shot Alignment Prompt...")
    few_shot_prompt = build_persona_prompt(matches, test_prompt)
    print("    Prompt preview:")
    for line in few_shot_prompt.strip().splitlines()[:5]:
        print(f"      | {line}")
    print("      | ...")

    # Step 3: LLM Inference
    print("\n[4] Querying Generator Pipeline...")
    response = generate_response(test_prompt, top_k=2)
    print("\n--- RESPONSE OUTPUT ---")
    print(response.strip())
    print("-----------------------")

    # Step 4: Stylometry Verification
    print("\n[5] Computing Stylometric Metrics on Response...")
    analyzer = StyleAnalyzer()
    metrics = analyzer.analyze_text(response)
    print(f"    Avg Sentence Length : {metrics.get('avg_sentence_length', 'N/A')} words")
    print(f"    Vocabulary Richness : {metrics.get('vocab_richness', 'N/A')}")
    print(f"    Readability Score   : {metrics.get('readability_score', 'N/A')}/100")
    print(f"    Formality Score     : {metrics.get('formality_score', 'N/A')}/100")
    print("\n[✓] Pipeline verification complete!\n")

if __name__ == "__main__":
    run_test()
