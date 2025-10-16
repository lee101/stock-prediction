#!/bin/bash

echo "Starting TensorBoard for RL Trading Agent logs..."
echo "================================================"
echo ""
echo "Logs directory: ./traininglogs/"
echo ""
echo "TensorBoard will be available at: http://localhost:6006"
echo ""
echo "Press Ctrl+C to stop TensorBoard"
echo ""
echo "================================================"

tensorboard --logdir=./traininglogs --bind_all