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
    """Run a quick subset of tests for development"""
    cmd = [
        sys.executable, "-m", "pytest",
        "test_database.py::TestEventDatabase::test_save_event",
        "test_model.py::TestAttentionFusion::test_forward_pass",
        "test_detection.py::TestPreprocessFrames::test_preprocess_frames_rgb_only",
        "test_api.py::TestAPIEndpoints::test_root_endpoint",
        "test_utils.py::TestSecureFilename::test_normal_filename",
        "-v",
        "--tb=line"
    ]
    return run_command(cmd, "Running quick test suite")


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
    
    missing_files = []
    for file in test_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing test files: {', '.join(missing_files)}")
        return False
    else:
        print("✅ All test files found")
        return True


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
    
    # Check if test files exist
    if not check_test_files():
        print("\n❌ Some test files are missing. Please ensure all test files are present.")
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
    elif args.command == "parallel":
        success = run_parallel_tests()
    elif args.command == "html":
        success = generate_html_report()
    elif args.command == "check":
        success = check_test_files()
    
    if success:
        print(f"\n🎉 {args.command.title()} completed successfully!")
        return 0
    else:
        print(f"\n💥 {args.command.title()} failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())