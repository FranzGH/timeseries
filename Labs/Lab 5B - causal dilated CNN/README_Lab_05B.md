# Lab 5B — Forecasting with causal and dilated CNNs

**Engineering MSc · Lecture 5, slides 99–262 · 120-minute core**

This lab follows Lab 5A on basic temporal convolution and pooling. Students predict a sensor response ten steps ahead in a controlled system with a 32-sample delay. The task makes the relationship between available history, receptive field, and output selection visible.

## Files

| File | Use |
|---|---|
| `lab05b_causal_dilated_cnn_student.ipynb` | Student notebook: commented code, experiments, questions, and exit ticket; outputs cleared |
| `lab05b_causal_dilated_cnn_instructor.ipynb` | Same code with discussion answers and outputs from the numerical exercises and baselines |
| `Instructor_Guide_Lab_05B.md` | Lesson timing, expected conclusions, interpretation cautions, and assessment rubric |
| `requirements.txt` | Python dependencies, including Keras 3 with the PyTorch backend |

All code is supplied. No dataset download or GPU is required. Optional WaveNet-style and multiscale activities add about 20–35 minutes of teaching time, excluding any additional training time.

## Run the lab

Use Python 3.11 or 3.12. Extract the complete ZIP and open a terminal in its directory. Creating a dedicated virtual environment is recommended.

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\activate` on Windows Command Prompt, or `source .venv/bin/activate` on macOS/Linux. Then install and start Jupyter:

```bash
python -m pip install -r requirements.txt
python -m jupyter lab
```

Open the student notebook and run cells in order. The setup cell selects the PyTorch backend **before** importing Keras. Restart the kernel if Keras was already imported with another backend. Packages can also be installed from a notebook with `%pip install -r requirements.txt`; restart the kernel afterward.

Keep the default settings for the initial comparison. The optional architecture section may be skipped entirely. `RUN_EXTRA_TRAINING = False` builds and inspects the optional models without training them. Training duration depends on the computer; the lesson schedule is a teaching allocation, not a measured runtime guarantee.

## Learning outcomes

- Calculate receptive fields, counting every convolution and accounting for stride and dilation.
- Distinguish a wide receptive-field span from complete coverage of its intermediate samples.
- Test causality by changing later observations while comparing earlier outputs.
- Compare residual CNNs with equal parameter counts and different accessible histories.
- Explain final-position selection, global average pooling, and per-time-step dense outputs.
- Evaluate forecasts against simple baselines using independent train, validation, and test runs.

The synthetic system is deliberately linear. Ridge regression is an appropriate strong baseline and may outperform the neural networks. The exercise evaluates architectural reasoning, not whether deep learning wins.

## Preparation and validation status

The numerical portions were executed with Python 3.12.14, NumPy 2.3.5, and scikit-learn 1.8.0. Nine code cells ran successfully, including data generation, receptive-field calculations, the NumPy causality test, and baseline evaluation. Four plots are embedded in the instructor notebook. Every code cell passed a Python syntax check.

**The six Keras/PyTorch-dependent cells were not executed.** These packages were unavailable in the preparation environment, and package downloads were blocked. Neural construction, backend assertions, training, and neural evaluation therefore require a local run before classroom delivery. Their notebook outputs are intentionally empty; no neural performance result is claimed. This is not a fully executed instructor notebook.

Verified baseline results with the supplied fixed seeds, in the original response units:

| Model | Validation RMSE | Test RMSE |
|---|---:|---:|
| Persistence | 1.2872 | 1.2317 |
| Ridge regression | 0.04739 | 0.04741 |
| Known-system reference | 0.04594 | 0.04583 |

The known-system reference uses the generator's true coefficients and past commands. It is an analytical reference, not a learned competitor. These numbers describe this synthetic benchmark only.

Before teaching, run the instructor notebook from a fresh kernel on the classroom environment. Verify the built-in shape, parameter-count, and causality assertions; record the measured neural results; and adjust the time allowance if needed. If installation is unavailable during class, Sections 1–5 remain usable with NumPy, Matplotlib, and scikit-learn.
