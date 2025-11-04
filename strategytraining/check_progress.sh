#!/bin/bash
# Quick progress checker for strategy dataset collection

echo "================================"
echo "Strategy Collection Progress"
echo "================================"
echo ""

# Check if progress file exists
if [ -f "strategytraining/datasets/collection_progress.json" ]; then
    echo "📊 Current Progress:"
    python3 -c "
import json
with open('strategytraining/datasets/collection_progress.json', 'r') as f:
    data = json.load(f)
    completed = data['completed_count']
    total = data['total_symbols']
    pct = (completed / total) * 100
    print(f'  Completed: {completed}/{total} symbols ({pct:.1f}%)')
    print(f'  Remaining: {total - completed} symbols')
    print(f'  Last updated: {data[\"last_updated\"][:19]}')
    print(f'')
    print(f'  Completed symbols: {', '.join(data[\"completed_symbols\"][:10])}...')
"
else
    echo "⚠️  No progress file found yet"
fi

echo ""
echo "🔍 Collection Process:"
ps aux | grep "[r]un_full_collection.py" | awk '{print "  Status: Running (PID " $2 ")"}'
if [ $? -ne 0 ]; then
    echo "  Status: Not running"
fi

echo ""
echo "📝 Recent Log Activity:"
tail -n 5 strategytraining/collection.log 2>/dev/null | sed 's/^/  /'

echo ""
echo "================================"
echo "Commands:"
echo "  tail -f strategytraining/collection.log  # Watch live log"
echo "  .venv/bin/python strategytraining/check_progress.py  # Detailed progress"
echo "================================"
