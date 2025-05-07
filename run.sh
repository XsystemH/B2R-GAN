action = $1

if [ "$action" = "train" ]; then
  echo "Training the model..."
  python scripts/train.py --config configs/default.yaml
elif [ "$action" = "eval" ]; then
  echo "Evaluating the model..."
  python scripts/eval.py --config configs/default.yaml
elif [ "$action" = "demo" ]; then
  echo "Running demo..."
  python scripts/demo.py --config configs/default.yaml
else
  echo "Invalid action. Use 'train', 'eval', or 'demo'."
fi