# BDH Sequence Model: Sim-to-Real Adaptation

A research repository investigating the role of online synaptic plasticity in fast adaptation to changing environment dynamics. We utilize the **BDH (Biologically Derived Hebbian)** sequence architecture, which uses Fast Weights (a dynamic $\sigma$ state) to adapt its own internal weights in real-time during inference. 

We test this architecture on its ability to solve the **Sim-to-Real gap**, where an agent trained in simulation must rapidly adapt when the real-world physics unexpectedly change mid-flight.

---

## 📁 Repository Structure
- `src/`: The core BDH architecture (`bdh.py`, `policy_models.py`).
- `pendulum/`: The classic Pendulum experiments, featuring a sudden mid-flight gravity shift.
- `lander/`: The advanced LunarLander experiments, featuring a catastrophic mid-flight thruster failure.

---

## 🚀 Experiment 1: Pendulum Gravity Switch (`pendulum/`)
We start with the classic `Pendulum-v1` environment. Midway through the rollout, the gravity suddenly shifts from $g=10$ to $g=20$.

By cloning an omniscient **Privileged Expert** into the BDH model using **DAgger**, the BDH architecture learns to dynamically infer the new gravity from its physical history. The moment the gravity switches, the BDH model's Fast Weights rewire to compensate, perfectly catching the pendulum and preventing it from falling over.

**To view live:**
```bash
cd pendulum
python run_live.py
```

---

## 🚀 Experiment 2: LunarLander Thruster Failure (`lander/`)
We scaled the architecture up to the much harder continuous control problem: `LunarLanderContinuous-v3`.

We engineered a **mid-flight catastrophic failure**: At timestep 60 (right in the middle of rapid descent), the left side-thruster loses 80% of its power. If the lander tries to use it normally, it induces an uncontrollable, deadly asymmetric spin.

**The Pipeline:**
1. **Train Privileged Expert:** We trained a DDPG expert with explicit knowledge of the thruster's health. It learned to land perfectly even when the thruster was broken by dynamically relying on the main engine to counter the spin.
2. **DAgger Distillation:** We cloned the Expert into the unprivileged BDH sequence model. The BDH model is blind to the thruster health—it must deduce the failure entirely from the unexpected rotational momentum it experiences.

**Results:**
When the thruster blows out, an untrained baseline will wildly spin into the ground and score a massive penalty (~ -500). The BDH model successfully uses its Fast Weight memory to detect the failure, instantly fight the spin, and stabilize into a soft crash or hover (-36), demonstrating incredibly powerful online adaptation.

**To view live:**
```bash
cd lander
python run_live_lander.py
```
