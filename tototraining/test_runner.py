#!/usr/bin/env python3
"""
Test runner and utility script for Toto retraining system tests.
Provides convenient commands to run different test suites.
"""

import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Optional
import json


class TestRunner:
    """Test runner for Toto retraining system"""
    
    def __init__(self, test_dir: Path = None):
        self.test_dir = test_dir or Path(__file__).parent
        self.test_files = self._discover_test_files()
    
    def _discover_test_files(self) -> List[Path]:
        """Discover all test files"""
        return list(self.test_dir.glob("test_*.py"))
    
    def run_unit_tests(self, verbose: bool = True) -> int:
        """Run unit tests"""
        cmd = [
            sys.executable, "-m", "pytest",
            "-m", "unit",
            "--tb=short",
            "-v" if verbose else "-q"
        ]
        return subprocess.call(cmd, cwd=self.test_dir)
    
    def run_integration_tests(self, verbose: bool = True) -> int:
        """Run integration tests"""
        cmd = [
            sys.executable, "-m", "pytest", 
            "-m", "integration",
            "--tb=short",
            "-v" if verbose else "-q"
        ]
        return subprocess.call(cmd, cwd=self.test_dir)
    
    def run_performance_tests(self, verbose: bool = True) -> int:
        """Run performance tests"""
        cmd = [
            sys.executable, "-m", "pytest",
            "-m", "performance",
            "--runperf",
            "--tb=short",
            "-v" if verbose else "-q"
        ]
        return subprocess.call(cmd, cwd=self.test_dir)
    
    def run_regression_tests(self, verbose: bool = True) -> int:
        """Run regression tests"""
        cmd = [
            sys.executable, "-m", "pytest",
            "-m", "regression", 
            "--tb=short",
            "-x",  # Stop on first failure for regression tests
            "-v" if verbose else "-q"
        ]
        return subprocess.call(cmd, cwd=self.test_dir)
    
    def run_data_quality_tests(self, verbose: bool = True) -> int:
        """Run data quality tests"""
        cmd = [
            sys.executable, "-m", "pytest",
            "-m", "data_quality",
            "--tb=short",
            "-v" if verbose else "-q"
        ]
        return subprocess.call(cmd, cwd=self.test_dir)
    
    def run_fast_tests(self, verbose: bool = True) -> int:
        """Run fast tests (excluding slow ones)"""
        cmd = [
            sys.executable, "-m", "pytest",
            "-m", "not slow",
            "--tb=short",
            "-v" if verbose else "-q"
        ]
        return subprocess.call(cmd, cwd=self.test_dir)
    
    def run_specific_test(self, test_file: str, test_name: str = None, verbose: bool = True) -> int:
        """Run a specific test file or test function"""
        target = test_file
        if test_name:
            target += f"::{test_name}"
        
        cmd = [
            sys.executable, "-m", "pytest",
            target,
            "--tb=short", 
            "-v" if verbose else "-q"
        ]
        return subprocess.call(cmd, cwd=self.test_dir)
    
    def run_all_tests(self, verbose: bool = True, include_slow: bool = False) -> int:
        """Run all tests"""
        cmd = [sys.executable, "-m", "pytest"]
        
        if not include_slow:
            cmd.extend(["-m", "not slow"])
        
        cmd.extend([
            "--tb=short",
            "-v" if verbose else "-q"
        ])
        
        return subprocess.call(cmd, cwd=self.test_dir)
    
    def run_with_coverage(self, output_dir: str = "htmlcov") -> int:
        """Run tests with coverage reporting"""
        try:
            import pytest_cov
        except ImportError:
            print("pytest-cov not installed. Install with: uv pip install pytest-cov")
            return 1
        
        cmd = [
            sys.executable, "-m", "pytest",
            "--cov=.",
            f"--cov-report=html:{output_dir}",
            "--cov-report=term-missing",
            "--cov-fail-under=70",
            "--tb=short"
        ]
        return subprocess.call(cmd, cwd=self.test_dir)
    
    def validate_test_environment(self) -> bool:
        """Validate test environment setup"""
        print("Validating test environment...")
        
        # Check required Python packages
        required_packages = [
            'pytest', 'torch', 'numpy', 'pandas', 'psutil'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                print(f"✓ {package} available")
            except ImportError:
                print(f"✗ {package} missing")
                missing_packages.append(package)
        
        # Check test files
        print(f"\nFound {len(self.test_files)} test files:")
        for test_file in self.test_files:
            print(f"  - {test_file.name}")
        
        # Check configuration files
        config_files = ['pytest.ini', 'conftest.py']
        for config_file in config_files:
            config_path = self.test_dir / config_file
            if config_path.exists():
                print(f"✓ {config_file} found")
            else:
                print(f"✗ {config_file} missing")
        
        if missing_packages:
            print(f"\nMissing packages: {', '.join(missing_packages)}")
            print("Install with: uv pip install " + " ".join(missing_packages))
            return False
        
        print("\n✅ Test environment validation passed!")
        return True
    
    def list_tests(self, pattern: str = None) -> int:
        """List available tests"""
        cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q"]
        
        if pattern:
            cmd.extend(["-k", pattern])
        
        return subprocess.call(cmd, cwd=self.test_dir)
    
    def run_dry_run(self) -> int:
        """Run tests in dry-run mode to check test discovery"""
        cmd = [
            sys.executable, "-m", "pytest",
            "--collect-only",
            "--tb=no"
        ]
        return subprocess.call(cmd, cwd=self.test_dir)
    
    def create_test_report(self, output_file: str = "test_report.json") -> int:
        """Create detailed test report"""
        cmd = [
            sys.executable, "-m", "pytest",
            "--json-report",
            f"--json-report-file={output_file}",
            "--tb=short"
        ]
        
        try:
            result = subprocess.call(cmd, cwd=self.test_dir)
            print(f"Test report saved to: {output_file}")
            return result
        except FileNotFoundError:
            print("pytest-json-report not installed. Install with: uv pip install pytest-json-report")
            return 1


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Test runner for Toto retraining system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s unit                    # Run unit tests
  %(prog)s integration            # Run integration tests  
  %(prog)s performance            # Run performance tests
  %(prog)s regression             # Run regression tests
  %(prog)s fast                   # Run fast tests only
  %(prog)s all                    # Run all tests
  %(prog)s all --slow             # Run all tests including slow ones
  %(prog)s specific test_toto_trainer.py    # Run specific test file
  %(prog)s coverage               # Run with coverage report
  %(prog)s validate               # Validate test environment
  %(prog)s list                   # List all tests
  %(prog)s list --pattern data    # List tests matching pattern
        """
    )
    
    parser.add_argument(
        'command',
        choices=[
            'unit', 'integration', 'performance', 'regression', 
            'data_quality', 'fast', 'all', 'specific', 'coverage',
            'validate', 'list', 'dry-run', 'report'
        ],
        help='Test command to run'
    )
    
    parser.add_argument(
        'target',
        nargs='?',
        help='Target for specific test (file or file::test_name)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--quiet', '-q', 
        action='store_true',
        help='Quiet output'
    )
    
    parser.add_argument(
        '--slow',
        action='store_true',
        help='Include slow tests'
    )
    
    parser.add_argument(
        '--pattern', '-k',
        help='Pattern to filter tests'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file/directory for reports'
    )
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = TestRunner()
    
    # Set verbosity
    verbose = args.verbose and not args.quiet
    
    # Execute command
    if args.command == 'unit':
        exit_code = runner.run_unit_tests(verbose=verbose)
        
    elif args.command == 'integration':
        exit_code = runner.run_integration_tests(verbose=verbose)
        
    elif args.command == 'performance':
        exit_code = runner.run_performance_tests(verbose=verbose)
        
    elif args.command == 'regression':
        exit_code = runner.run_regression_tests(verbose=verbose)
        
    elif args.command == 'data_quality':
        exit_code = runner.run_data_quality_tests(verbose=verbose)
        
    elif args.command == 'fast':
        exit_code = runner.run_fast_tests(verbose=verbose)
        
    elif args.command == 'all':
        exit_code = runner.run_all_tests(verbose=verbose, include_slow=args.slow)
        
    elif args.command == 'specific':
        if not args.target:
            print("Error: specific command requires target argument")
            return 1
        
        if '::' in args.target:
            test_file, test_name = args.target.split('::', 1)
        else:
            test_file, test_name = args.target, None
            
        exit_code = runner.run_specific_test(test_file, test_name, verbose=verbose)
        
    elif args.command == 'coverage':
        output_dir = args.output or "htmlcov"
        exit_code = runner.run_with_coverage(output_dir)
        
    elif args.command == 'validate':
        success = runner.validate_test_environment()
        exit_code = 0 if success else 1
        
    elif args.command == 'list':
        exit_code = runner.list_tests(pattern=args.pattern)
        
    elif args.command == 'dry-run':
        exit_code = runner.run_dry_run()
        
    elif args.command == 'report':
        output_file = args.output or "test_report.json"
        exit_code = runner.create_test_report(output_file)
        
    else:
        print(f"Unknown command: {args.command}")
        exit_code = 1
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())