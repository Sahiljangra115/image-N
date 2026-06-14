import os
from rag import build_index, retrieve

def run_tests():
    dataset_path = "data/dataset"
    print("Building index from folder:", dataset_path)
    
    # 1. Build index
    # This will load bakery_intro.txt, recipes_and_allergies.docx, inventory_pricing.xlsx, safety_compliance.pdf
    chunks, index = build_index(dataset_path)
    print(f"Total chunks created: {len(chunks)}")
    
    # 2. Test queries
    test_queries = [
        ("What is the mission statement of Maya's Sweet Haven?", "bakery_intro.txt"),
        ("How many grams of chocolate are in the Chocolate Lava Cake?", "recipes_and_allergies.docx"),
        ("What is the price of Vegan Blueberry Muffin?", "inventory_pricing.xlsx"),
        ("Where is the First Aid Kit located in the kitchen?", "safety_compliance.pdf")
    ]
    
    print("\n--- Running Retrieval Tests ---")
    all_passed = True
    for query, expected_source in test_queries:
        print(f"\nQuery: '{query}'")
        top_chunks = retrieve(query, chunks, index, k=3)
        print("Top Chunks retrieved:")
        for idx, chunk in enumerate(top_chunks):
            print(f"  [{idx + 1}] {chunk[:120].replace('\n', ' ')}...")
        
        # Verify the top retrieved chunk is from the expected file
        top_chunk = top_chunks[0]
        actual_source = "unknown"
        if f"[source: {expected_source}]" in top_chunk:
            actual_source = expected_source
            print(f"✅ Success! Top chunk matches expected source: {expected_source}")
        else:
            # Check if it contains the source tag in any of the top 3
            found_in_top_3 = False
            for c in top_chunks:
                if f"[source: {expected_source}]" in c:
                    found_in_top_3 = True
                    break
            if found_in_top_3:
                print(f"⚠️ Partial Success: Expected source '{expected_source}' was found in top 3, but not as the 1st result.")
            else:
                print(f"❌ Failure! Expected source '{expected_source}' not found in top retrieved chunks.")
                all_passed = False
                
    if all_passed:
        print("\n🎉 All retrieval tests passed successfully!")
    else:
        print("\n❌ Some retrieval tests failed.")

if __name__ == "__main__":
    run_tests()
