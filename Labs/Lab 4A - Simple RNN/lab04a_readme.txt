LAB 4A -- SimpleRNN Forecasting
Based on Lecture 4: slides 1-95 and 230-242.

Contents:
  lab04a_rnn_student.ipynb            Guided beginner student lab
  lab04a_rnn_teacher_solutions.ipynb  Complete commented code and instructor answers
  lab04a_rnn_intro_slides.pptx        Ten editable slide briefings

Topics:
  * Train/validation/test split by whole independent recordings
  * One-step forecasting: persistence, Ridge, SimpleRNN
  * Tensor shapes and recurrent parameter sharing
  * Illustrative gradient vanishing, explosion, and gradient clipping
  * Direct multi-horizon 10-step forecasts
  * Sequence-to-sequence training with a last-step evaluation metric

Dependencies: Python, numpy, matplotlib, scikit-learn, keras>=3, torch.
The notebook selects the Keras PyTorch backend to run without TensorFlow.
If missing, install Keras and PyTorch before running the notebook.
CPU is sufficient. No external data required.

The notebooks are executable from top to bottom and contain no missing-code TODOs;
student activities ask for interpretations and optional small modifications.
Numerical model results may differ slightly between machine environments.
