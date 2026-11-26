# Lab 5A — Find the Fault: Temporal CNNs

**Audience:** Beginner Engineering MSc students, after Lecture 5 slides 1–98.  
**Duration:** ~120 minutes (plus optional 15-minute exploration).  
**Topic boundaries:** Conv1D filters/feature maps; kernels, stride, padding; local and global pooling; introduction to the receptive-field idea. **No causal/dilated convolution, TCN or WaveNet yet.**

## Files

- `lab05a_temporal_cnn_student.ipynb` — the guided lab. The code is heavily annotated and **runs end-to-end as supplied**; students are invited to modify settings and explain results.
- `lab05a_temporal_cnn_teacher_executed.ipynb` — the same lab with instructor solutions, teaching notes, plots and verified execution results.
- `lab05a_temporal_cnn_intro_slides.pptx` — 11 editable introductory briefing slides.

## Setup (CPU; no external dataset)

Using a local Python 3.11+ environment, for example:

```bash
python -m pip install "keras>=3" torch numpy matplotlib scikit-learn notebook ipykernel
jupyter notebook
```

Open the student notebook and select a Python kernel. It sets `KERAS_BACKEND=torch` **before** importing Keras. If you already imported Keras using another backend in that kernel, restart the kernel before starting the notebook. GPU is not required.

## Teaching schedule

| Time | Activity | Learning outcome |
|---|---|---|
| 0–5 min | Setup | Keras backend and reproducibility |
| 5–20 min | Hand filter and feature map | Weight sharing and ReLU |
| 20–35 min | Conv1D shape experiments | Kernel/stride/padding/filters |
| 35–50 min | Pooling examples | Max/average and local/global differences |
| 50–65 min | Create and inspect sensor windows | Temporal motif and independent splits |
| 65–90 min | Train baseline and CNN | Validation/test separation, feature learning |
| 90–105 min | Max versus mean global pooling | Short-event detection trade-off |
| 105–115 min | Feature-map visualization and downsampling | Interpretability and timing loss |
| 115–120 min | Exit ticket | Explain the main design decisions |

An additional 10–15 minutes can be used to test another kernel size.

## Dataset and task

The dataset consists of independent, generated univariate vibration windows of length 128 at a nominal sampling frequency of 100 Hz. Classes are *smooth* versus *ringing* events, inserted at randomized times into noisy background vibration. There are 900 training, 200 validation and 200 test windows. The event energies are approximately matched to discourage solving the task purely from amplitude. These are **synthetic demonstration data, not validated industrial fault signatures**.

## Expected results from one executed instructor run

- Three-feature logistic regression baseline: test accuracy **0.645**.
- CNN with global max pooling: test accuracy **1.000**.
- CNN with global average pooling: test accuracy **0.970**.

These numbers reflect this controlled dataset and training seed, not a general accuracy ranking. Results may differ across library versions and hardware. The purpose is to connect architecture and feature representations to empirical observations.

**Teaching cautions:** Global pooling operates across the *whole* temporal window, but it does not itself learn event ordering or precise separation. The Conv1D feature maps preserve time positions (with alignment dependent on padding), while global pooling discards those positions. Max pooling is useful for isolated peaks but does not guarantee robustness. Do not make model choices using test results; choose using the validation set and evaluate the held-out test set once.

## Student deliverables

Submit the executed student notebook plus a short written answer (around 200–300 words total or one slide) covering: kernel and stride, difference between local and global pooling, a train/validation/test comparison, and why a CNN for whole-window classification may not be appropriate for detecting exact event onset times.
