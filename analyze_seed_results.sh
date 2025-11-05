#!/bin/bash
# Quick analysis of seed test results

echo "========================================="
echo "SEED TEST RESULTS SUMMARY"
echo "========================================="
echo ""

if [ ! -f /tmp/seed_test.log ]; then
    echo "Test not complete yet"
    exit 1
fi

# Extract key findings
echo "SEED VARIANCE (Uncompiled):"
grep -A 3 "Uncompiled seed variance:" /tmp/seed_test.log

echo ""
echo "SEED VARIANCE (Compiled):"
grep -A 3 "Compiled seed variance:" /tmp/seed_test.log

echo ""
echo "SEED-BY-SEED COMPARISON:"
grep -A 6 "Seed " /tmp/seed_test.log | grep -E "Seed |MAE difference:|Sample correlation:"

echo ""
echo "KEY FINDINGS:"
grep -A 10 "KEY FINDINGS" /tmp/seed_test.log | tail -8

echo ""
echo "CONCLUSION:"
grep -A 6 "HYPOTHESIS" /tmp/seed_test.log
