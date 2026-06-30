# BDH Adaptation for Control

A research experiment investigating the role of online synaptic plasticity in fast adaptation to changing environment dynamics.

---

## Hypothesis

**Primary Hypothesis ($H_1$):**
The online synaptic plasticity of the BDH (Dragon Hatchling) architecture—governed by the fast-weight state $\sigma$ that updates during inference via a Hebbian rule—enables the model to adapt to mid-rollout changes in environment dynamics faster than:
1.  **Frozen-$\sigma$ BDH:** The same trained BDH model with its synaptic-state update clamped off at test time.
2.  **Transformer:** A parameter-matched Transformer with a fair context window (adapting via in-context learning).

**Null / Kill Condition:**
If the live-$\sigma$ BDH does not outperform the frozen-$\sigma$ BDH in recovery speed/error after the dynamics switch, the plasticity effect is unsupported. In this event, the experiment will be halted and the findings documented.

---

## Experimental Design

The experiment is divided into two phases using the `Pendulum-v1` environment:

### Phase A: Open-Loop Next-State Prediction (Core)
*   **Training:** Train models to predict the next state $s_{t+1}$ given the history of states and actions $(s_{\le t}, a_{\le t})$ under **dynamics regime A** (e.g., standard pole mass/length/gravity).
*   **Testing:** Run evaluation rollouts where the dynamics switch from **regime A $\rightarrow$ regime B** (e.g., altered gravity or pole mass) mid-episode. No gradient updates are performed at test time.
*   **Evaluation:** Measure the prediction error recovery curve and the time-to-recover.

### Phase B: Closed-Loop Control (Stretch)
*   **Training:** Train a reinforcement learning expert (using SAC/PPO) on regime A, and behavior-clone the expert's policy into a continuous-input BDH model.
*   **Testing:** Run the policy in a closed-loop environment where the dynamics switch mid-episode.
*   **Evaluation:** Measure the recovery in episode return and stabilization time.

---

## Model Baselines

To ensure a rigorous comparison, all models are evaluated under the same parameter and training-budget constraints:
*   **Live-$\sigma$ BDH:** Active online Hebbian updates during inference.
*   **Frozen-$\sigma$ BDH (Ablation):** Clamped $\sigma$ during inference to isolate the effect of plasticity from the static weights.
*   **Param-matched Transformer:** Evaluated with a sufficiently long context window to allow for in-context adaptation.
*   **GRU / MLP:** Lower-bound baselines to calibrate performance.

---

## Metrics

*   **Prediction Error Curve:** MSE between predicted and actual next states over time.
*   **Time-to-Recover:** The number of timesteps after the dynamics switch until the prediction error returns to within 110% of the pre-switch baseline.
*   **Area Under the Recovery Curve (AURC):** The cumulative prediction error post-switch.

---

## Project Structure & Roadmap

*   **`bdh.py`**: Model architecture.
*   **`train.py`**: Toy training script (original language modeling setup).
*   **Milestone M0 (Current):** Environment setup and stock model sanity checks.
*   **Milestone M1:** Adapting BDH for continuous state-action inputs and next-state regression.
*   **Milestone M2:** Implementing the mid-rollout switch harness and establishing the live-$\sigma$ baseline.
*   **Milestone M3:** Implementing the frozen-$\sigma$ ablation and baseline models.
*   **Milestone M4:** Multi-seed runs, aggregating results, and plotting figures.
