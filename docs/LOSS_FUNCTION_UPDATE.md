# Q&A System: Training Loss Implementation Summary

## Problem Identified

The previous training was converging due to unsupervised losses (entropy + L2 regularization) but **not actually learning to predict correct answer positions**. The model was achieving low loss by being confident *somewhere*, not necessarily at the *correct* position.

## Root Cause

The `compute_span_loss()` function was not using the `start_positions` and `end_positions` parameters passed from the batch data. This meant:
- No supervision signal about correct answer locations  
- Model could minimize loss by predicting confidently anywhere
- Inference produced random-seeming answers with low confidence

## Solution Implemented

Replaced the loss function with a **supervised cross-entropy loss** that:

1. **Creates soft target distributions**: For each target position, generates a Gaussian-shaped probability distribution centered at that position
   - Gaussian width (sigma=2.0) focuses supervision near target but allows nearby positions
   - Accounts for tokenization ambiguities and partial-token matches

2. **Computes cross-entropy penalty**: 
   ```
   Loss = -sum(target_dist * log(model_pred))
   ```
   - Directly penalizes assigning low probability to target regions
   - Fully differentiable (all operations preserve gradients)
   - Numerically stable (uses log-softmax via clamp + log)

3. **Handles batch dimension properly**:
   - Each batch item gets its own target distribution
   - Matrix operations efficiently compute loss for all positions simultaneously

## Key Technical Details

- **Position indices**: [batch_size, seq_length] matrix of position-to-position distances
- **Soft targets**: Gaussian weights `exp(-(distance^2)/sigma^2)` normalized to probability distributions
- **Cross-entropy computation**: `-sum(target_weights * log(model_probs))`
- **Numerical stability**: Clamp probabilities to 1e-10 minimum before log to prevent -inf

## Expected Behavior Changes

### Before (Unsupervised):
- Training loss decreases rapidly but doesn't correlate with accuracy
- Model converges to low loss without learning meaningful patterns
- Inference: Random-seeming answers with 0.1-0.2 confidence

### After (Supervised):
- Training loss provides actual supervision signal
- Model should gradually learn to concentrate probability at correct positions
- Validation accuracy should improve from ~0% (currently random)
- Inference: More coherent answers reflecting actual document positions

## Next Steps

### 1. **Validate Loss Stability** (IMMEDIATE)
```bash
cargo build --release && cargo run --release -- train 2>&1 | head -50
```
Monitor first 3 batches for:
- `Loss: NaN` or `Loss: inf` → Loss function has numerical issues, needs fallback
- `Loss: 2.5-5.0` range (typical cross-entropy on classified positions) → ✓ Good
- Loss decreasing over iterations → ✓ Learning

### 2. **Monitor Convergence Pattern**
Compare to previous runs:
- **Bad sign**: Loss jumps to NaN after batch 1-2 (numerical instability)
- **Good sign**: Loss starts ~3-5, gradually decreases to 0.5-1.0 over 10 epochs
- **Excellent**: Validation accuracy improves from 0% to 10-30%

### 3. **If NaN/Inf Reoccurs**
Fallback approaches (ranked by complexity):
1. Reduce sigma to 0.5 (make targets sharper, less smoothing)
2. Add small L2 regularization back: `loss + 0.0001 * logits.norm()`
3. Use straight-through estimator for argmax-based losses
4. Switch to separate regression head for positions (model architecture change)

### 4. **Hyperparameter Tuning** (IF loss is stable)
Based on observed training curves:
- **Sigma too large** (loss decreasing slowly): Reduce sigma → 0.5-1.0
- **Sigma too small** (loss volatile/noisy): Increase sigma → 3.0-5.0  
- **Loss stuck at plateau**: Increase learning rate (currently 0.0001) → try 0.0005
- **Loss diverges**: Reduce learning rate → 0.00001

### 5. **Evaluate Improvements**
After 10 epochs training with new loss:
```bash
cargo run --release -- infer \
  --doc-path "data/calendar_2026.docx" \
  --question "What is scheduled for January 20, 2026?" \
  --model-path "model_epoch_10"
```

Compare to previous result:
- **Before**: "schools" (0.1136 confidence)
- **Expected**: Actual event from calendar with 0.5+ confidence

## Code Changes

**File**: `src/training.rs`
**Function**: `compute_span_loss()`
- **Removed**: Entropy + L2 regularization (unsupervised)
- **Added**: Gaussian-weighted cross-entropy (supervised)
- **Lines**: ~25-80 of function

## Caveats and Considerations

1. **Tokenization alignment**: Target positions are token indices, but answer spans can start/end mid-token
   - Gaussian soft targets handle this by allowing nearby positions
   - Some tolerance built in via sigma parameter

2. **Position clamping**: Answers beyond seq_length boundaries get clamped to valid range
   - This could create misleading supervision for truncated answers
   - Consider filtering training data to exclude truncated examples

3. **Batch accuracy metric**: Current validation uses binary accuracy (top-1 match)
   - Should improve validation reporting to show F1/EM (exact match) scores  
   - Currently reported as ~0% because exact argmax rarely matches exactly

4. **Model capacity**: Current model is very small (64 dims, 1 layer)
   - This could be a bottleneck even with perfect loss function
   - Consider experimenting with increased capacity (d_model=128, n_layers=2) after loss works

## Summary

This change introduces **proper supervision** to match the task: the model now receives a clear signal about where answer positions should be concentrated. This should transform training from an unsupervised convergence process to actual discriminative learning of position predictions.

The next run will reveal:
1. Whether the loss function is numerically stable (first success criterion)
2. Whether supervised learning improves accuracy (second success criterion)
3. What fine-tuning is needed for optimal performance (third iteration)
