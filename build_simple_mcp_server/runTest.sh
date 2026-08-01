#!/bin/bash

# This script runs a specific pytest test case.
# It's used by the Code Lab's checking mechanism.

TEST_CASE=$1

# Find the file containing the test case (search for function name with test_ prefix)
TEST_FILE=$(grep -rl "def test_${TEST_CASE}" tests/ || grep -rl "${TEST_CASE}" tests/)

if [ -z "$TEST_FILE" ]; then
  echo "AssertionFailedError: Test case $TEST_CASE not found!"
  exit 1
fi

# Run the specific test (pytest function syntax: file::function_name)
python3 -m pytest "${TEST_FILE}::test_${TEST_CASE}"