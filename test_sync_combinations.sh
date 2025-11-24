#!/bin/bash
# Test different synchronization combinations to find minimal requirements

set -e

PRELOAD="LD_PRELOAD=/usr/local/fbcode/platform010/lib/libcublasLt.so:/usr/local/fbcode/platform010/lib/libcublas.so"
NUM_REQUESTS=100
TIMEOUT=400

echo "=========================================="
echo "EAGLE Synchronization Minimization Tests"
echo "=========================================="
echo ""
echo "Testing different sync combinations with $NUM_REQUESTS requests"
echo "Each test has a ${TIMEOUT}s timeout"
echo ""

# Backup current eagle.py
cp vllm/v1/spec_decode/eagle.py vllm/v1/spec_decode/eagle.py.backup

function restore_eagle() {
    cp vllm/v1/spec_decode/eagle.py.backup vllm/v1/spec_decode/eagle.py
}

function run_test() {
    local test_name="$1"
    local log_file="$2"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Test: $test_name"
    echo "Log: $log_file"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    CUDA_LAUNCH_BLOCKING=0 $PRELOAD timeout $TIMEOUT \
        python benchmark_eagle.py --num-requests $NUM_REQUESTS \
        > "$log_file" 2>&1

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        # Check if all requests completed
        if grep -q "Requests processed:.*$NUM_REQUESTS" "$log_file"; then
            echo "✅ PASS - All $NUM_REQUESTS requests completed"
            return 0
        else
            echo "❌ FAIL - Did not complete all requests"
            return 1
        fi
    elif [ $exit_code -eq 124 ]; then
        echo "⏱️  TIMEOUT - Test exceeded ${TIMEOUT}s"
        return 1
    else
        echo "❌ FAIL - Exit code: $exit_code"
        # Show last error
        echo "Last 10 lines of log:"
        tail -10 "$log_file"
        return 1
    fi
}

# Test 1: Baseline - All 3 syncs (should PASS)
echo ""
echo "Test 1: BASELINE (all 3 syncs - entry, intra-loop, exit)"
restore_eagle
run_test "Baseline (all 3 syncs)" "/tmp/sync_test_baseline.log"
BASELINE_RESULT=$?

# Test 2: Only intra-loop sync
echo ""
echo "Test 2: ONLY INTRA-LOOP sync (remove entry + exit)"
restore_eagle
# Remove entry sync (line 233)
sed -i '233d' vllm/v1/spec_decode/eagle.py
# Remove exit sync (line 516, now 515 after previous deletion)
sed -i '515,517d' vllm/v1/spec_decode/eagle.py
run_test "Only intra-loop" "/tmp/sync_test_only_intraloop.log"
ONLY_INTRALOOP_RESULT=$?

# Test 3: Entry + intra-loop sync (NO exit)
echo ""
echo "Test 3: ENTRY + INTRA-LOOP sync (no exit)"
restore_eagle
# Remove only exit sync (line 516-517)
sed -i '514,517d' vllm/v1/spec_decode/eagle.py
run_test "Entry + intra-loop" "/tmp/sync_test_entry_intraloop.log"
ENTRY_INTRALOOP_RESULT=$?

# Test 4: Intra-loop + exit sync (NO entry)
echo ""
echo "Test 4: INTRA-LOOP + EXIT sync (no entry)"
restore_eagle
# Remove entry sync (line 233)
sed -i '230,233d' vllm/v1/spec_decode/eagle.py
run_test "Intra-loop + exit" "/tmp/sync_test_intraloop_exit.log"
INTRALOOP_EXIT_RESULT=$?

# Test 5: Only entry + exit (NO intra-loop) - expect FAIL
echo ""
echo "Test 5: ENTRY + EXIT sync (no intra-loop) - EXPECT FAIL"
restore_eagle
# Remove intra-loop sync (line 506-509)
sed -i '506,509d' vllm/v1/spec_decode/eagle.py
run_test "Entry + exit (no intra-loop)" "/tmp/sync_test_entry_exit.log"
ENTRY_EXIT_RESULT=$?

# Restore original
restore_eagle
rm vllm/v1/spec_decode/eagle.py.backup

# Summary
echo ""
echo "=========================================="
echo "RESULTS SUMMARY"
echo "=========================================="
echo ""
echo "Test 1 - Baseline (all 3):            $([ $BASELINE_RESULT -eq 0 ] && echo '✅ PASS' || echo '❌ FAIL')"
echo "Test 2 - Only intra-loop:             $([ $ONLY_INTRALOOP_RESULT -eq 0 ] && echo '✅ PASS' || echo '❌ FAIL')"
echo "Test 3 - Entry + intra-loop:          $([ $ENTRY_INTRALOOP_RESULT -eq 0 ] && echo '✅ PASS' || echo '❌ FAIL')"
echo "Test 4 - Intra-loop + exit:           $([ $INTRALOOP_EXIT_RESULT -eq 0 ] && echo '✅ PASS' || echo '❌ FAIL')"
echo "Test 5 - Entry + exit (no intraloop): $([ $ENTRY_EXIT_RESULT -eq 0 ] && echo '✅ PASS' || echo '❌ FAIL')"
echo ""

# Recommendation
echo "=========================================="
echo "RECOMMENDATION"
echo "=========================================="
echo ""

if [ $ENTRY_INTRALOOP_RESULT -eq 0 ]; then
    echo "✅ Minimal sync requirement: ENTRY + INTRA-LOOP"
    echo "   Exit sync can be removed without issues"
elif [ $ONLY_INTRALOOP_RESULT -eq 0 ]; then
    echo "✅ Minimal sync requirement: ONLY INTRA-LOOP"
    echo "   Both entry and exit syncs can be removed!"
elif [ $INTRALOOP_EXIT_RESULT -eq 0 ]; then
    echo "✅ Minimal sync requirement: INTRA-LOOP + EXIT"
    echo "   Entry sync can be removed"
else
    echo "⚠️  All 3 syncs appear necessary"
    echo "   Cannot reduce further without failures"
fi

echo ""
echo "Logs available at: /tmp/sync_test_*.log"
