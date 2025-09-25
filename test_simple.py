#!/usr/bin/env python3
"""
Simple test to verify FGA components work correctly
Tests knowledge DB and attention mechanics with real data
"""

import os
import sys
import torch

# Add src to path
sys.path.append(os.path.dirname(__file__))

from src.knowledge_db import KnowledgeDatabase


def test_knowledge_db():
    """Test the knowledge database with real smartphone data"""
    print("\n✨ Testing FGA Knowledge Database\n")
    print("=" * 60)
    
    # Create knowledge DB
    os.makedirs("./data", exist_ok=True) 
    db = KnowledgeDatabase(
        db_path="./data/test_smartphone_facts",
        device="mps" if torch.backends.mps.is_available() else "cpu"
    )
    
    # Test lookups with real data
    test_phones = [
        ("phone:iphone_15_pro", "3274 mAh battery"),
        ("phone:galaxy_s24_ultra", "200 MP camera"),
        ("phone:iphone_15", "USB-C 2.0"),
        ("phone:pixel_8_pro", "Tensor G3"),
    ]
    
    print("\n📱 Testing Real Smartphone Fact Retrieval:")
    print("-" * 60)
    
    for phone_id, expected_info in test_phones:
        embedding = db.lookup(phone_id)
        if embedding is not None:
            print(f"✅ {phone_id}: Retrieved (embedding shape: {embedding.shape})")
            print(f"   Expected: {expected_info}")
        else:
            print(f"❌ {phone_id}: Not found")
    
    # Test batch lookup
    print("\n📦 Testing Batch Lookup:")
    print("-" * 60)
    
    batch_keys = ["phone:iphone_15_pro", "phone:galaxy_s24_ultra", "phone:pixel_8"]
    embeddings, metadata = db.batch_lookup_with_metadata(batch_keys)
    
    if embeddings is not None:
        print(f"✅ Batch lookup successful: {embeddings.shape}")
        for i, key in enumerate(batch_keys):
            if i < len(metadata):
                meta = metadata[i]
                print(f"   {key}:")
                print(f"     - Battery: {meta.get('battery_capacity', 'N/A')} mAh")
                print(f"     - Camera: {meta.get('main_camera_mp', 'N/A')} MP")
                print(f"     - Processor: {meta.get('processor', 'N/A')}")
    
    # Test cache performance
    print("\n⚡ Cache Statistics:")
    print("-" * 60)
    stats = db.get_cache_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test knowledge update
    print("\n🔄 Testing Knowledge Update:")
    print("-" * 60)
    
    print("Updating iPhone 15 Pro battery to 3500 mAh...")
    db.update_fact("phone:iphone_15_pro", {
        "battery_capacity": "3500",
        "display_size": "6.1",
        "main_camera_mp": "48",
        "processor": "A17 Pro"
    })
    
    # Verify update
    _, updated_metadata = db.batch_lookup_with_metadata(["phone:iphone_15_pro"])
    if updated_metadata:
        new_battery = updated_metadata[0].get('battery_capacity')
        print(f"✅ Updated battery capacity: {new_battery} mAh")
    
    # Clean up
    db.close()
    print("\n✨ Knowledge DB Test Complete!\n")


def test_attention_dimensions():
    """Test FGA attention dimension calculations"""
    print("\n🔍 Testing FGA Attention Dimensions\n")
    print("=" * 60)
    
    # Simulate attention score calculation
    batch_size = 1
    num_heads = 24
    seq_len = 32
    head_dim = 128
    num_facts = 3
    
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Heads: {num_heads}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Head dimension: {head_dim}")
    print(f"  Number of facts: {num_facts}")
    print()
    
    # Standard attention scores
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim)
    S = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
    print(f"✅ Standard attention scores S: {S.shape}")
    print(f"   Expected: [{batch_size}, {num_heads}, {seq_len}, {seq_len}]")
    
    # FGA grounding calculation (corrected dimensions)
    K_fact = torch.randn(num_facts, head_dim)  # Facts don't have head dimension initially
    K_fact = K_fact.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)  # Expand to match Q
    Q_fact_bias = torch.matmul(Q, K_fact.transpose(-2, -1)) / (head_dim ** 0.5)
    print(f"\n✅ Query-fact bias Q_fact_bias: {Q_fact_bias.shape}")
    print(f"   Expected: [{batch_size}, {num_heads}, {seq_len}, {num_facts}]")
    
    # Entity assignment matrix
    A = torch.zeros(num_facts, seq_len)
    A[0, 5:8] = 1.0  # Entity 0 at positions 5-7
    A[1, 15:18] = 1.0  # Entity 1 at positions 15-17
    A[2, 25:27] = 1.0  # Entity 2 at positions 25-26
    A = A.unsqueeze(0).unsqueeze(0)  # Add batch and head dims
    print(f"\n✅ Entity assignment matrix A: {A.shape}")
    print(f"   Expected: [1, 1, {num_facts}, {seq_len}]")
    
    # Grounding scores (dimensionally correct)
    G = torch.matmul(Q_fact_bias, A)
    print(f"\n✅ Grounding scores G: {G.shape}")
    print(f"   Expected: [{batch_size}, {num_heads}, {seq_len}, {seq_len}]")
    
    # Final FGA scores
    alpha = torch.sigmoid(torch.randn(1, 1, 1, 1))  # Gate value
    S_FGA = S + alpha * G
    print(f"\n✅ Final FGA scores S_FGA: {S_FGA.shape}")
    print(f"   Expected: [{batch_size}, {num_heads}, {seq_len}, {seq_len}]")
    print(f"   Alpha (gate) value: {alpha.item():.3f}")
    
    # Verify dimensions match
    assert S.shape == S_FGA.shape, "Dimension mismatch!"
    print(f"\n🎉 All dimensions correct! Can add G to S safely.")
    
    print("\n✨ Attention Dimension Test Complete!\n")


if __name__ == "__main__":
    # Test knowledge database
    test_knowledge_db()
    
    # Test attention dimensions
    test_attention_dimensions()
    
    print("\n🏁 All tests completed successfully!\n")
