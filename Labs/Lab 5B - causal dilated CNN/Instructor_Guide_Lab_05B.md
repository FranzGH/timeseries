# Instructor guide — Lab 5B

**Forecasting with causal and dilated CNNs**

This activity covers the second part of `05 CNNs(1).pptx`, following Lab 5A. Students should already understand a temporal tensor `[batch, time, channels]`, local convolution, basic pooling, and a train/validation/test split. They need not write network code from scratch.

## Lesson schedule

| Section | Activity | Minutes |
|---|---|---:|
| 0 | Setup and prediction task | 5 |
| 1 | Receptive-field calculations and coverage | 15 |
| 2 | Suffix-perturbation causality experiment | 15 |
| 3 | Final-position and average readouts | 10 |
| 4 | Delayed system, independent runs, and target alignment | 15 |
| 5 | Persistence, Ridge, and analytical reference | 10 |
| 6 | Build matched residual CNNs and inspect assertions | 15 |
| 7 | Train and compare validation forecasts | 10 |
| 8 | Inspect prediction-head tensor shapes | 5 |
| 9 | Evaluate the frozen comparison on held-out runs | 10 |
| 10 | Exit ticket | 10 |
| **Total core** | | **120** |

The optional section on WaveNet-style gating and parallel multiscale branches adds 20–35 minutes, plus any optional training. Its default setting builds and inspects models without fitting them. Core timing should be checked against the actual teaching hardware: neural training has not been timed or executed in the preparation environment.

## Experiment and expected reasoning

The supplied generator is:

\[
y_t=0.5y_{t-1}+0.8u_{t-32}+\epsilon_t,
\qquad u_t\sim\mathcal N(0,1),\quad \epsilon_t\sim\mathcal N(0,0.04^2).
\]

Each sample supplies 64 historical observations of command and response, ending at time `t`, and predicts `y[t+1:t+11]`. The commands entering these ten future responses occur at historical lags 31 through 22. These are outside a final-position receptive field of 17 samples, but inside one of 61 samples.

The three core networks share 6,490 parameters, channel widths, convolution counts, initial kernel values, optimization settings, and training limits. They vary in dilation schedule and readout. Initial-weight equality and parameter counts are asserted by the supplied code, but those backend assertions still need execution in the teaching environment.

| Question | Expected conclusion |
|---|---|
| Why does a 64-sample input not guarantee access to all 64 samples? | The graph connecting a particular output to the input determines its usable history. |
| Why is the dilated span 61, not 31? | Each of four blocks has two serial size-three convolutions: `1 + 2 × 2 × (1+2+4+8) = 61`. |
| Can dilation create gaps? | Yes. Four size-three convolutions at dilation four span 33 samples but connect only nine positions. |
| What does the suffix-perturbation test establish? | Per-time-step outputs of the tested causal computation do not depend on later input positions. It does not validate the complete data pipeline. |
| Is `same` padding always wrong for forecasting? | No. An encoder of a fully observed historical window can use centered filters without seeing observations beyond the forecast origin. Online outputs at every position require a stricter causal interpretation. |
| Why try the final representation? | It is anchored to the forecast origin and avoids explicitly averaging representations at different positions. Its accessible history still depends on the backbone. |
| Why might global averaging work? | It combines features throughout the observed window. The encoder can learn useful features before pooling. It is a valid alternative, not an inherently incorrect forecasting head. |
| Does pooling preserve all timing information? | No. The averaging operation cannot distinguish rearrangements of identical feature vectors. A learned encoder can encode order before averaging, so this is not a universal claim about the entire network. |
| What does an unpooled `Dense(10)` output? | `[B,L,10]`: ten values per position. Applying it to `[B,C]` instead produces one `[B,10]` forecast per sequence. |
| Should the TCN beat Ridge? | No such claim is warranted. The process is linear, and Ridge receives the full history. |

The short model's final output cannot access the independently sampled commands needed for the specified future horizons. The dilated model has the structural capacity to access them. Successful optimization and a particular neural RMSE do not follow from receptive-field arithmetic alone.

For the pooled model, “61” describes the maximum span of each individual feature vector before pooling. The final prediction aggregates features across the whole window. Do not present its final prediction as having a 61-sample restriction.

## Controls and interpretation

- Independent simulated runs are assigned to train, validation, and test before overlapping windows are made. Scaling is fitted on raw training runs only.
- The known-system reference uses true generator parameters and the same historical inputs. It demonstrates the future-noise limit; it is not a trained baseline.
- Validation data support model choices and early stopping. Test results are inspected only after those choices are frozen.
- Per-run RMSE variation is descriptive. It is not a confidence interval; overlapping windows within a run are correlated.
- The first comparison uses one seed and one synthetic process. Any observed architecture ranking is limited to that setting.
- Keep the delay at least as large as the prediction horizon when using the supplied analytical reference, so every required command is already in the historical window. Also ensure the history is long enough to contain those commands.
- The optional networks demonstrate gating, skip connections, and parallel scales. They are not complete reproductions of WaveNet or InceptionTime, and they are not matched in parameter count to the core networks.

Useful prompts during training: ask students to calculate the earliest needed command index, explain the difference between a structural dependency and a learned dependency, or predict what changes when the readout averages all temporal positions.

## Assessment

Students submit their executed notebook and a 250–400-word response to the exit ticket. Suggested rubric:

| Criterion | Points |
|---|---:|
| Correct receptive-field calculation and interpretation of coverage | 3 |
| Correct causality and leakage reasoning | 2 |
| Correct output-shape and readout explanation | 2 |
| Evidence-based comparison with baselines and acknowledgment of limitations | 3 |
| **Total** | **10** |

Reward explanations supported by the experiment, even when a neural model trains poorly. Do not award points for obtaining a predetermined model ranking.

## Delivery checklist

The supplied instructor notebook contains executed numerical demonstrations and baseline results. **Its six Keras/PyTorch cells have not been executed.** Packages were unavailable and installation downloads were blocked in the preparation environment. Run the notebook from a fresh kernel before distributing measured neural results or using the lab live.

Check that all three core networks have 6,490 parameters, initialization comparisons pass, fixed Keras convolutions agree with the NumPy calculations, and the causal backbones pass the perturbation checks. Record the actual training duration. If a model fails to learn sufficiently within the default budget, discuss that outcome or adjust the training budget using validation data before the classroom comparison; do not select changes using test performance.

## Lecture mapping

| Slides | Lab content |
|---|---|
| 99–124 | Receptive fields |
| 125–144 | Causality |
| 145–168 | Dilation |
| 169–193; 250–253 | Residual TCN |
| 194–212; 254–258 | Optional WaveNet-style model |
| 213–227; 245–249; 261–262 | Tasks, tensor shapes, and readouts |
| 228–244; 259–260 | Optional multiscale model |
