import time
import numpy as np
from Neural_Net_Utils import flattened_board_to_tensor, prepare_neural_net_instance
from constant_strings import GAME_TICTACTOE, MCTS_NN

def test_prediction_speed():
    """Test the speed difference between regular and XLA-optimized predictions."""
    
    print("🚀 Testing XLA Performance Optimization")
    print("=" * 50)
    
    # Initialize neural network
    neural_net = prepare_neural_net_instance(GAME_TICTACTOE, size=4, ai_type=MCTS_NN)
    
    # Create test board states (simulating MCTS simulations)
    test_states = []
    for i in range(1000):  # Simulate 1000 MCTS simulations
        # Create a random board state
        board_str = "".join(np.random.choice([".", "X", "O"], size=16))
        test_states.append(board_str)
    
    print(f"Testing with {len(test_states)} board states...")
    
    # Test 1: Regular predict (old method)
    print("\n📊 Testing Regular Predict Method:")
    start_time = time.time()
    
    for board_str in test_states:
        tensor = flattened_board_to_tensor(board_str, GAME_TICTACTOE)[None, ...]
        _ = neural_net.model.predict(tensor, verbose=0)
    
    regular_time = time.time() - start_time
    print(f"⏱️  Regular predict time: {regular_time:.3f} seconds")
    
    # Test 2: Fast predict (XLA optimized)
    print("\n📊 Testing XLA-Optimized Fast Predict Method:")
    start_time = time.time()
    
    for board_str in test_states:
        tensor = flattened_board_to_tensor(board_str, GAME_TICTACTOE)[None, ...]
        _ = neural_net.fast_predict(tensor)
    
    fast_time = time.time() - start_time
    print(f"⏱️  Fast predict time: {fast_time:.3f} seconds")
    
    # Test 3: Batch predict (most efficient)
    print("\n📊 Testing Batch Predict Method:")
    start_time = time.time()
    
    tensors = []
    for board_str in test_states:
        tensor = flattened_board_to_tensor(board_str, GAME_TICTACTOE)
        tensors.append(tensor)
    
    batch_tensor = np.array(tensors)
    _ = neural_net.batch_predict(batch_tensor)
    
    batch_time = time.time() - start_time
    print(f"⏱️  Batch predict time: {batch_time:.3f} seconds")
    
    # Calculate speedups
    speedup_fast = regular_time / fast_time
    speedup_batch = regular_time / batch_time
    
    print("\n🎯 Performance Results:")
    print(f"✅ Fast predict speedup: {speedup_fast:.1f}x faster")
    print(f"✅ Batch predict speedup: {speedup_batch:.1f}x faster")
    
    if speedup_fast > 1.5:
        print("🎉 XLA compilation is working well!")
    else:
        print("⚠️  XLA compilation may not be providing significant speedup")
    
    print(f"\n💡 For 1500 MCTS simulations:")
    print(f"   Regular: ~{regular_time * 1.5:.1f} seconds")
    print(f"   Fast:    ~{fast_time * 1.5:.1f} seconds")
    print(f"   Batch:   ~{batch_time * 1.5:.1f} seconds")

def test_mcts_cache_performance():
    """Test the MCTS caching performance."""
    
    print("\n🔍 Testing MCTS Caching Performance")
    print("=" * 50)
    
    from Mcts import Mcts
    from tictactoe_variant import Tictactoe
    
    # Create a test game instance
    game = Tictactoe(size=4, ai_type=MCTS_NN, simulation_mode=True)
    
    # Initialize MCTS
    mcts = Mcts(game, game)
    
    # Test repeated predictions on same board state
    board_str = "".join(str(cell) for row in game.board for cell in row)
    
    print(f"Testing repeated predictions on same board state...")
    
    # Test without cache (simulated)
    start_time = time.time()
    for _ in range(100):
        tensor = flattened_board_to_tensor(board_str, GAME_TICTACTOE)[None, ...]
        _ = mcts.neural_net.fast_predict(tensor)
    no_cache_time = time.time() - start_time
    
    # Test with cache
    start_time = time.time()
    for _ in range(100):
        _ = mcts._get_cached_prediction(board_str)
    cache_time = time.time() - start_time
    
    cache_speedup = no_cache_time / cache_time
    
    print(f"⏱️  No cache time: {no_cache_time:.3f} seconds")
    print(f"⏱️  With cache time: {cache_time:.3f} seconds")
    print(f"✅ Cache speedup: {cache_speedup:.1f}x faster")

if __name__ == "__main__":
    test_prediction_speed()
    test_mcts_cache_performance()
    
    print("\n🎯 Recommendations:")
    print("1. Use fast_predict() for individual predictions")
    print("2. Use batch_predict() when possible for multiple states")
    print("3. MCTS caching will automatically avoid redundant predictions")
    print("4. XLA compilation provides the biggest speedup for repeated calls") 