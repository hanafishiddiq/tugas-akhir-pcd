"""Run invariant tests and emit an optional code-bound receipt."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import unittest

from readiness import CORE_TESTS, code_hashes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt",type=Path)
    args = parser.parse_args()
    before = code_hashes()
    suite = unittest.defaultTestLoader.loadTestsFromNames([Path(f).stem for f in CORE_TESTS])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    after = code_hashes()
    if not result.wasSuccessful() or before != after or result.testsRun==0 or result.skipped:
        return 1
    if args.receipt:
        report = {
            "suite":list(CORE_TESTS), "tests_run":result.testsRun,"tests_failed":0,
            "code_hashes":after,"tested_at_utc":datetime.now(timezone.utc).isoformat(),
            "tests_cover":"Synthetic contract invariants and frozen package diagnostics",
        }
        args.receipt.write_text(json.dumps(report,indent=2)+"\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
