#!/usr/bin/env python3
"""
Test runner for the agentic AI system
"""

import sys
import os
import subprocess

def run_test(test_file):
    """Run a specific test file"""
    print(f"\n🧪 Running {test_file}...")
    print("=" * 50)
    
    try:
        result = subprocess.run([sys.executable, f"tests/{test_file}"], 
                              capture_output=True, text=True, cwd=os.getcwd())
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running {test_file}: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🚀 Running All Tests for Agentic AI System")
    print("=" * 60)
    
    # List of test files to run
    test_files = [
        "test_basic_functionality.py",
        "test_workflow.py",
        "test_api.py"
    ]
    
    results = {}
    
    for test_file in test_files:
        success = run_test(test_file)
        results[test_file] = success
        
        if success:
            print(f"✅ {test_file} PASSED")
        else:
            print(f"❌ {test_file} FAILED")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_file, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_file}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

def main():
    """Main function"""
    if len(sys.argv) > 1:
        # Run specific test
        test_file = sys.argv[1]
        if test_file in ["basic", "functionality"]:
            success = run_test("test_basic_functionality.py")
        elif test_file in ["workflow", "workflows"]:
            success = run_test("test_workflow.py")
        elif test_file in ["api", "openai"]:
            success = run_test("test_api.py")
        else:
            print(f"Unknown test: {test_file}")
            print("Available tests: basic, workflow, api")
            sys.exit(1)
    else:
        # Run all tests
        success = run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
