# Lab 04B — LSTM and GRU (beginner MSc)

**Prerequisite:** Lab 04A — SimpleRNN for temporal forecasting.

**Time:** approximately 120 minutes, with a further 15–20 minutes of optional extension.

## Files

- `lab04b_lstm_gru_student.ipynb` — guided notebook with five TODO coding exercises, explanations and questions.
- `lab04b_lstm_gru_teacher_executed.ipynb` — complete worked solutions with outputs and plots already embedded. It was executed successfully with Keras 3 / PyTorch backend.
- `lab04b_lstm_gru_teacher_solutions.ipynb` — identical solutions, without stored outputs, for a clean instructor rerun.
- `lab04b_lstm_gru_intro_slides.pptx` — 11 editable introductory slides with diagrams and speaker notes.

## Setup

Python 3.10+ and the following libraries are sufficient:

```bash
pip install keras torch numpy matplotlib scikit-learn notebook
```

Start Jupyter and open the student notebook. Alternatively, run in a supported notebook environment with these dependencies installed. The notebook sets `KERAS_BACKEND="torch"` before importing Keras, just like the preceding SimpleRNN lab. If changing the Keras backend, restart the kernel and choose the backend **before** `import keras`.

**CPU only:** no GPU or data download is necessary. The dataset is generated in the notebook.

## Exercises and suggested timings

1. **LSTM memory arithmetic (15 min)** — cell state versus hidden state, forget/input/output gates, elementwise calculations and output-gate visualization.
2. **GRU update (10 min)** — reset and update gate roles, single-state candidate/old-state mixing.
3. **Memory-retention intuition (10 min)** — constant forget-gate demonstration; the values are illustrative, not learned gate measurements.
4. **Forecasting dataset and model structure (25 min)** — same 50-sample past and direct 10-step future as Lab 04A, recording-level splits, state shapes and parameter counts.
5. **Forecast models and baselines (35 min)** — persistence, Ridge, SimpleRNN, LSTM and GRU; identical splits and training budget.
6. **Performance interpretation (20 min)** — overall MSE/MAE, per-horizon MAE, example forecasts, parameter counts, and validation learning curves.
7. **Exit ticket (5 min)** — a short engineering recommendation with evidence.

## Important teaching cautions

- The provided gate values in the first exercises are a conceptual demonstration, not actual activations retrieved from a trained LSTM/GRU.
- The GRU toy equation uses the explicit convention `h_new = z * h_old + (1-z) * candidate`. Literature may use a complementary definition of `z`.
- Keras `GRU(reset_after=True)` uses two bias vectors; its parameter count differs from some simplified equations found in slides and books.
- Split complete recordings before constructing windows; otherwise overlapping windows can leak nearly identical samples across splits.
- Choose model architecture and settings with **validation**, then report test results. Results can vary by device, package version and random initialization.
- A gated architecture is not necessarily more accurate or economical than a linear model on a relatively simple periodic dataset.
- Keras 3 applies `Dense(10)` along the last axis of a rank-3 tensor without requiring `TimeDistributed(Dense(10))`.

## Verification performed

The instructor notebook completed successfully with no execution errors using Keras 3 / PyTorch backend, including 6 embedded figures. The PPTX was rendered and passed the presentation overflow check.
