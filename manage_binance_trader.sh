#!/bin/bash
# Manage Binance trading bot

SESSION="binance-trader"

case "$1" in
    start)
        if tmux has-session -t $SESSION 2>/dev/null; then
            echo "❌ Bot is already running"
            exit 1
        fi
        echo "🚀 Starting Binance trader..."
        tmux new-session -d -s $SESSION "cd /home/lee/code/stock && bash /tmp/start_binance_trader.sh"
        sleep 2
        echo "✅ Bot started in tmux session '$SESSION'"
        echo "To view: tmux attach -t $SESSION"
        ;;
    
    stop)
        if ! tmux has-session -t $SESSION 2>/dev/null; then
            echo "❌ Bot is not running"
            exit 1
        fi
        echo "🛑 Stopping Binance trader..."
        tmux kill-session -t $SESSION
        echo "✅ Bot stopped"
        ;;
    
    restart)
        $0 stop
        sleep 2
        $0 start
        ;;
    
    status)
        if tmux has-session -t $SESSION 2>/dev/null; then
            echo "✅ Bot is RUNNING"
            echo ""
            echo "Last output:"
            tmux capture-pane -t $SESSION -p | tail -20
        else
            echo "❌ Bot is NOT running"
        fi
        ;;
    
    view)
        if ! tmux has-session -t $SESSION 2>/dev/null; then
            echo "❌ Bot is not running"
            exit 1
        fi
        echo "📺 Attaching to bot (Ctrl+B then D to detach)..."
        sleep 1
        tmux attach -t $SESSION
        ;;
    
    logs)
        if [ ! -f strategy_state/binanceexp1-solusd-metrics.csv ]; then
            echo "❌ No metrics log found"
            exit 1
        fi
        tail -f strategy_state/binanceexp1-solusd-metrics.csv
        ;;
    
    balance)
        cd /home/lee/code/stock
        source .venv313/bin/activate
        python -c "from src.binan import binance_wrapper; import json; print(json.dumps(binance_wrapper.get_account_value_usdt(), indent=2))"
        ;;
    
    *)
        echo "Usage: $0 {start|stop|restart|status|view|logs|balance}"
        echo ""
        echo "Commands:"
        echo "  start    - Start the trading bot"
        echo "  stop     - Stop the trading bot"
        echo "  restart  - Restart the trading bot"
        echo "  status   - Check if bot is running and show recent output"
        echo "  view     - Attach to bot terminal (Ctrl+B then D to exit)"
        echo "  logs     - Tail the metrics log"
        echo "  balance  - Check account balance"
        exit 1
        ;;
esac
