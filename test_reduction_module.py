"""
Test script for reduction.
"""

import numpy as np
from reduction import (
    create_reducer,
    get_available_reducers,
    PCAReducer,
    UMAPReducer,
    TSNEReducer
)


def test_factory():
    """
    Test factory function.
    """
    print("\n" + "="*60)
    print("TEST 1: Factory Function")
    print("\n" + "="*60)

    try:
        # Get available algorithms
        algorithms = get_available_reducers()
        print(f"\n  Available algorithms: {algorithms}")

        # Create each algorithm
        for algo in algorithms:
            reducer = create_reducer(algo, n_components=10)
            print(f"   Created {algo}: {reducer}")
        
        print("\n TEST 1 PASSED")
        return True
    except Exception as e:
        print(f"\n TEST 1 FAILED: {e}")
        return False

def run_all_tests():
    """
    Run all tests.
    """
    print("="*60)
    print("REDUCTION MODULE TEST SUITE")
    print("="*60)
    
    results = []

    results.append(("Factory Function", test_factory()))

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for name, passed in results:
        status = " PASSED" if passed else "FAILED"
        print(f"{name:30s} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED!")
        print("="*60)
        print("\nThe reduction module is working correctly.")
    else:
        print("SOME TESTS FAILED")
        print("="*60)
        print("\nPlease check the error messages above.")
    
    return all_passed

if __name__ =="__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)