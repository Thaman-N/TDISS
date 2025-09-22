"""
Test runner script for the violence detection backend.

This script provides easy ways to run different test suites and generate reports.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and return success status"""
    if description:
        print(f"\n{'='*60}")
        print(f"{description}")
        print(f"{'='*60}")
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print(f"✅ {description or 'Command'} completed successfully")
        return True
    else:
        print(f"❌ {description or 'Command'} failed with exit code {result.returncode}")
        return False


def install_test_dependencies():
    """Install test dependencies using pip"""
    print("📦 Installing test dependencies via pip...")
    cmd = [sys.executable, "-m", "pip", "install", "-r", "test_requirements.txt"]
    return run_command(cmd, "Installing test dependencies")


def run_unit_tests():
    """Run unit tests only"""
    cmd = [
        sys.executable, "-m", "pytest",
        "test_database.py",
        "test_model.py", 
        "test_detection.py",
        "test_utils.py",
        "-v",
        "--tb=short"
    ]
    return run_command(cmd, "Running unit tests")


def run_api_tests():
    """Run API tests only"""
    cmd = [
        sys.executable, "-m", "pytest",
        "test_api.py",
        "-v",
        "--tb=short"
    ]
    return run_command(cmd, "Running API tests")


def run_all_tests():
    """Run all tests"""
    cmd = [
        sys.executable, "-m", "pytest",
        "-v",
        "--tb=short",
        "--durations=10"
    ]
    return run_command(cmd, "Running all tests")


def run_tests_with_coverage():
    """Run tests with coverage report"""
    cmd = [
        sys.executable, "-m", "pytest",
        "--cov=main",
        "--cov=model",
        "--cov=torch_detection",
        "--cov-report=html",
        "--cov-report=term-missing",
        "-v"
    ]
    return run_command(cmd, "Running tests with coverage")


def run_quick_tests():
    """Run a quick subset of tests for development - UPDATED for optimized architecture"""
    cmd = [
        sys.executable, "-m", "pytest",
        # Database tests
        "test_database.py::TestEventDatabase::test_save_event",
        
        # Updated model tests (removed AttentionFusion, added new components)
        "test_model.py::TestSE3D::test_forward_pass",
        "test_model.py::TestOptimizedX3DViolenceDetector::test_model_initialization",
        "test_model.py::TestSimpleConcatenation::test_concatenation_fusion",
        
        # Detection pipeline tests
        "test_detection.py::TestPreprocessFrames::test_preprocess_frames_rgb_only",
        "test_detection.py::TestPredictViolence::test_predict_violence_non_violent",
        "test_detection.py::TestExtractFrames::test_extract_frames_success",
        
        # API tests (if they exist)
        "test_api.py::TestAPIEndpoints::test_root_endpoint",
        
        # Utility tests (if they exist)
        "test_utils.py::TestSecureFilename::test_normal_filename",
        
        "-v",
        "--tb=line"
    ]
    return run_command(cmd, "Running quick test suite (optimized architecture)")


def run_model_tests_only():
    """Run only model-related tests"""
    cmd = [
        sys.executable, "-m", "pytest",
        "test_model.py",
        "-v",
        "--tb=short"
    ]
    return run_command(cmd, "Running model tests only")


def run_detection_tests_only():
    """Run only detection pipeline tests"""
    cmd = [
        sys.executable, "-m", "pytest",
        "test_detection.py",
        "-v",
        "--tb=short"
    ]
    return run_command(cmd, "Running detection pipeline tests only")


def run_parallel_tests():
    """Run tests in parallel for faster execution"""
    cmd = [
        sys.executable, "-m", "pytest",
        "-n", "auto",  # Use all available CPUs
        "-v"
    ]
    return run_command(cmd, "Running tests in parallel")


def generate_html_report():
    """Generate HTML test report"""
    cmd = [
        sys.executable, "-m", "pytest",
        "--html=test_report.html",
        "--self-contained-html",
        "-v"
    ]
    return run_command(cmd, "Generating HTML test report")


def check_test_files():
    """Check if all test files exist"""
    test_files = [
        "test_database.py",
        "test_model.py", 
        "test_detection.py",
        "test_api.py",
        "test_utils.py",
        "pytest.ini",
        "test_requirements.txt"
    ]
    
    existing_files = []
    missing_files = []
    
    for file in test_files:
        if Path(file).exists():
            existing_files.append(file)
        else:
            missing_files.append(file)
    
    print(f"✅ Found test files: {', '.join(existing_files)}")
    
    if missing_files:
        print(f"⚠️  Missing optional test files: {', '.join(missing_files)}")
        print("   (These will be skipped in test runs)")
    
    # Only require core test files
    required_files = ["test_model.py", "test_detection.py"]
    missing_required = [f for f in required_files if not Path(f).exists()]
    
    if missing_required:
        print(f"❌ Missing required test files: {', '.join(missing_required)}")
        return False
    else:
        print("✅ All required test files found")
        return True


def run_core_tests():
    """Run only the core tests that should always exist"""
    existing_test_files = []
    
    # Check which test files actually exist
    test_files = ["test_model.py", "test_detection.py", "test_database.py", "test_api.py", "test_utils.py"]
    for test_file in test_files:
        if Path(test_file).exists():
            existing_test_files.append(test_file)
    
    if not existing_test_files:
        print("❌ No test files found!")
        return False
    
    cmd = [
        sys.executable, "-m", "pytest"
    ] + existing_test_files + [
        "-v",
        "--tb=short"
    ]
    
    return run_command(cmd, f"Running core tests: {', '.join(existing_test_files)}")


def run_architecture_tests():
    """Run tests specifically for the optimized architecture components"""
    cmd = [
        sys.executable, "-m", "pytest",
        # SE3D attention tests
        "test_model.py::TestSE3D",
        
        # Motion enhancement tests
        "test_model.py::TestMotionEnhancementModule",
        
        # Optimized model tests
        "test_model.py::TestOptimizedX3DViolenceDetector",
        
        # Simple concatenation tests
        "test_model.py::TestSimpleConcatenation",
        
        # Detection pipeline with new architecture
        "test_detection.py::TestIntegration",
        
        "-v",
        "--tb=short"
    ]
    return run_command(cmd, "Running optimized architecture tests")


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="Violence Detection Backend Test Runner")
    
    parser.add_argument(
        "command",
        choices=[
            "install",
            "unit",
            "api", 
            "all",
            "coverage",
            "quick",
            "core",
            "model",
            "detection", 
            "architecture",
            "parallel",
            "html",
            "check"
        ],
        help="Test command to run"
    )
    
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Skip dependency installation check"
    )
    
    args = parser.parse_args()
    
    print("Violence Detection Backend Test Runner")
    print("=====================================")
    print("🔧 Updated for Optimized X3D Architecture")
    print("")
    
    # Check if test files exist
    if not check_test_files():
        if args.command not in ["check", "install"]:
            print("\n❌ Required test files are missing. Please ensure test files are present.")
            return 1
    
    # Install dependencies if needed
    if not args.no_install and args.command != "install" and args.command != "check":
        if not Path("test_requirements.txt").exists():
            print("⚠️  test_requirements.txt not found, skipping dependency installation")
        else:
            print("🔧 Checking test dependencies...")
            # Try a simple import to see if pytest is available
            try:
                import pytest
                print("✅ Pytest is available")
            except ImportError:
                print("📦 Installing test dependencies...")
                if not install_test_dependencies():
                    return 1
    
    # Execute the requested command
    success = False
    
    if args.command == "install":
        success = install_test_dependencies()
    elif args.command == "unit":
        success = run_unit_tests()
    elif args.command == "api":
        success = run_api_tests()
    elif args.command == "all":
        success = run_all_tests()
    elif args.command == "coverage":
        success = run_tests_with_coverage()
    elif args.command == "quick":
        success = run_quick_tests()
    elif args.command == "core":
        success = run_core_tests()
    elif args.command == "model":
        success = run_model_tests_only()
    elif args.command == "detection":
        success = run_detection_tests_only()
    elif args.command == "architecture":
        success = run_architecture_tests()
    elif args.command == "parallel":
        success = run_parallel_tests()
    elif args.command == "html":
        success = generate_html_report()
    elif args.command == "check":
        success = check_test_files()
    
    if success:
        print(f"\n🎉 {args.command.title()} completed successfully!")
        
        # Show helpful next steps
        if args.command == "quick":
            print("\n💡 Next steps:")
            print("   • Run 'python run_tests.py all' for complete test suite")
            print("   • Run 'python run_tests.py architecture' for optimized model tests")
            print("   • Run 'python run_tests.py coverage' for coverage report")
        
        return 0
    else:
        print(f"\n💥 {args.command.title()} failed!")
        
        # Show helpful debugging tips
        if args.command == "quick":
            print("\n🔍 Try these debugging steps:")
            print("   • Run 'python run_tests.py check' to verify test files")
            print("   • Run 'python run_tests.py core' to run only existing tests")
            print("   • Check that test files match the updated architecture")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())