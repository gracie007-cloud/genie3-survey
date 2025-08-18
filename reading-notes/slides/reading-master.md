# Genie: Genie: Generative Interactive Environments

## 1. What is Genie?
* A large (≈11B params) foundation world model turning a **single prompt image** (text2img output / sketch / photo) into an **interactive, controllable 2D world**.
* Trained **only on unlabeled Internet videos** (platformer gameplay + robotics) – no actions, text, engine state.
* Interactivity via a **tiny discrete latent action space** (|A|=8) learned unsupervised (frame-by-frame control).
* Users or agents press latent “buttons” each timestep to advance the generated trajectory.
* Core joint idea:
	- Temporal-aware video tokenization.
	- Discrete latent action codebook capturing inter-frame change.
	- Autoregressive spatiotemporal dynamics model predicting next-frame tokens conditioned on chosen latent actions.

![Figure 1: High-Level Concept](./figures/figure1_hook-1.png)
*Figure 1 (High-Level Concept):* **A whole new world** — Genie converts a variety of different prompts into interactive, playable environments that can be easily created, stepped into, and explored. This is made possible via a latent action interface, learned fully unsupervised from Internet videos. On the right (in the original banner) are example generated steps for two latent actions. More examples: https://sites.google.com/view/genie-2024/home

## 2. Technical Contributions (Condensed)
* Paradigm: **Generative Interactive Environment** (playable, not passive video).
* **Unsupervised latent action discovery** with VQ codebook (|A|=8) over pixels.
* **Unified ST Transformer** blocks for tokenizer, LAM, dynamics (linear-ish scaling in T for spatial cost).
* **ST-ViViT tokenizer**: temporal-aware, more efficient & better than C-ViViT.
* **MaskGIT dynamics + action embeddings** for sharper controllable prediction.
* **ΔPSNR controllability metric** (difference vs random actions).
* **Scaling study** across params & batch (40M→10B dyn model) shows smooth gains.
* **Cross-domain generality** (platformers + robotics) without labeled actions.
* **Policy transfer demo**: latent→real action mapping enables few-shot BC.

## 3. Inference Inputs & Outputs (Use Case Focus)
This section describes ONLY the runtime (inference) interaction loop—how a user (or autonomous agent) supplies inputs and receives outputs from Genie.

> "A player first prompts the model with an image $x_1$ that serves as the initial frame."  
> "The player then specifies a discrete latent action $a_1$ to take by choosing any integer value within $[0,|A|)$."  
> "The dynamics model takes the frame tokens $z_1$ and corresponding latent action ... to predict the next frame tokens $z_2$."  
> "This process is repeated to generate the rest of the sequence ... while tokens are decoded into video frames."

These quotes (Inference: Action-Controllable Video Generation section) define the inference contract we model below.

### 3.2 Inference Inputs & Outputs

![Figure: Genie Inference](../../papers/genie-tex/figures/genie_inference.png)
*Figure (Genie Inference):* the prompt frame is tokenized, combined with the latent action taken by the user, and passed to the dynamics model for iterative generation. The predicted frame tokens are then decoded back to image space via the tokenizer's decoder.


Inputs:
* **Prompt Frame(s)**: One (or few) initial image(s). Can be:
	- Hand-drawn sketch
	- Text-to-image output
	- Real-world photo
	- In-domain gameplay frame
* **Latent Action Selection per Timestep**: Discrete integer `a_t ∈ {0,…,|A|-1}` (|A| typically = 8) chosen by:
	- Human player (manual exploration)
	- Learned policy / agent (automated control)
* **(Optional) Stopping Condition**: User stop, max frames, or environment-specific termination.

Outputs:
* **Next Frame (RGB)**: Decoded visual frame `x_{t+1}` for display.
* **Updated Internal State (Implicit)**: Appended token history `z_{1:t+1}` (not surfaced to user directly).
* **Interactive Trajectory**: Sequence of rendered frames forming a controllable rollout.

### 3.5 Activity Diagram
Source: `figures/inference-activity.puml`  
Rendered: ![Activity Diagram](./figures/InferenceActivity.svg)

## 4. Architecture Overview
![Figure 2: Genie Model Training Architecture](./figures/genie_architecture.png)
*Figure 2 (Genie model training — original caption):* **Genie model training:** Genie takes in $T$ frames of video as input, tokenizes them into discrete tokens $z$ via the video tokenizer, and infers the latent actions $\tilde{a}$ between each frame with the latent action model. Both are then passed to the dynamics model to generate predictions for the next frames in an iterative manner.

### 4.1 High-Level Overview
At training time Genie ingests raw frame sequences and splits responsibility across three tightly coupled components that all share the ST (spatial–temporal) Transformer design:
* **Video Tokenizer (VQ-VAE with ST attention)**: Converts each incoming frame (plus limited causal temporal context) into a grid of discrete tokens \(z_t\). Temporal awareness (causal attn over same spatial patch index) improves efficiency vs full 3D attention.

	![Tokenizer I/O](./figures/TokenizerIO.svg)

* **Latent Action Model (LAM)**: ST-transformer over prior frames plus next frame (pixels) produces latent actions \(\tilde{a}_{1:t}\), vector-quantized into a tiny discrete codebook (|A|=8) chosen for playability & controllability (paper lines 178–179, 248). Trained with VQ-VAE objective + reconstruction decoder (training-only); at inference encoder/decoder are discarded and user supplies discrete action index \(a_t\) whose embedding (codebook lookup) conditions dynamics. Pixel inputs outperform token inputs for controllability (lines 333–335).

	![LAM I/O](./figures/LAMIO.svg)

* **Dynamics Model (ST MaskGIT Transformer)**: Autoregressively predicts the *next frame tokens* conditioned on past tokens and the selected (or inferred during training) latent action embedding. Uses iterative masked token refinement for sharper, globally consistent predictions.

	![Dynamics I/O](./figures/DynamicsIO.svg)

---

### 4.2 Video Tokenizer Architecture (ST‑ViViT VQ‑VAE)
Figure: `tokenizer_architecture.png`

![Tokenizer Architecture](../../papers/genie-tex/figures/tokenizer_architecture.png)

Internal ST Transformer Block (as used inside the tokenizer and shared design across modules):

![ST Transformer Block](../../papers/genie-tex/figures/sttransformer.png)

Block: Spatial attn → causal temporal attn (same patch index) → FFN (residual + LN) for linear temporal scaling.

Essentials:
* Input: frame sequence.
* Stages: patch embed → spatial blocks → causal temporal attn → VQ bottleneck → (decoder for training only).
* Output: discrete token grid z_t.
* Why: efficient temporal modeling, discrete interface for masking & control, compact codebook.

Notes: small codebook balances fidelity/efficiency; bfloat16 + QK norm for stability.

### 4.3 Latent Action Model (LAM) Architecture
Figure: `LAM_architecture.png`

![LAM Architecture](../../papers/genie-tex/figures/LAM_architecture.png)

Essentials:
* Input: full pixel frame context \(x_{1:t}, x_{t+1}\) (raw frames; pixel input yields better controllability than token inputs per ablation).
* Stages: sequence ST encoder (causal over time) → latent action logits for all steps \(\tilde{a}_{1:t}\) → VQ codebook (|A|=8) → (training-only) reconstruction head.
* Output: discrete action index a_t + embedding ã_t (only embedding lookup used at inference).
* Why tiny codebook: forces semantic reuse, amplifies per-code influence (higher ΔPSNR).
* Losses: VQ (commitment + codebook) + reconstruction; diversity encouraged implicitly by small |A|.
* Inference: model skipped; controller supplies a_t directly.

### 4.4 Dynamics Model Architecture (ST MaskGIT Transformer)
Figure: `dynamics_architecture.png`

![Dynamics Architecture](../../papers/genie-tex/figures/dynamics_architecture.png)

Essentials:
* Input: past token grids z_{≤t} (each z_k is a discrete H×W grid of VQ token IDs produced by the video tokenizer for frame k) + current action embedding ã_t.
* Loop (per frame): mask init → iterative MaskGIT refinements (≈25) → additive action conditioning each step → spatial attn → temporal causal attn → FFN → logits → unmask until complete.
* Output: next token grid z_{t+1} (decoded to RGB via tokenizer decoder when needed).
* Why iterative: parallel token filling + global consistency vs strictly autoregressive order.
* Control: additive ã_t modulation yields immediate, smooth per-frame influence.
* Shared block: spatial attn → temporal causal attn → FFN (same ST pattern as tokenizer/LAM internals).

Mask (MaskGIT) concept:
* Placeholder token for unknown positions.
* Start high mask ratio (often 100%).
* Predict all positions in parallel; reveal (replace) high-confidence masks each iteration.
* Continue until no masks remain (produces final z_{t+1}).

## 5. Training Pipeline
1. Pretrain tokenizer (VQ-VAE) on clips (recon + commitment losses).
2. Jointly train LAM + dynamics:
	 * LAM: VQ loss + reconstruction from pixels; small codebook (|A|=8).
	 * Dynamics: cross-entropy on tokens with random masking rate U[0.5,1].
3. Inference: encode prompt frame → user selects action index each step → lookup embedding → predict next tokens → decode frame.

### 5.1 Detailed Component Training Flow

Data construction (base corpus):
* Collect raw Internet gameplay + robotics videos.
* Slice into fixed-length clips (e.g., 16–32 frames) with frame resizing & normalization.
* Maintain chronological order; no action labels required.

Stage A — Video Tokenizer Pretraining (frozen later):
* Input: RGB clip.
* Optimize reconstruction loss (L1 / perceptual) + VQ commitment & codebook update.
* Output: encoder + codebook + decoder; retain encoder + codebook for downstream, freeze weights.
* Dataset requirement: only frames (no pairing beyond temporal sequencing).

Intermediate dataset artifact:
* For each frame in corpus, store its discrete token grid z (can be on-the-fly to save storage if fast enough).

Stage B — Joint LAM + Dynamics:
* LAM branch inputs: (x_t, x_{t+1}) raw frames.
* LAM losses: reconstruction (predict x_{t+1}), VQ (commitment + codebook). Produces action code index a_t and embedding ã_t.
* Dynamics branch inputs: token sequences z_{≤t} (from frozen tokenizer) + corresponding latent action embeddings ã_{≤t-1}.
* Dynamics loss: cross-entropy over predicted next frame tokens with random masking schedule U[0.5,1] on target frame tokens to enable iterative refinement behavior.
* Parameter update: LAM + Dynamics (tokenizer encoder & decoder remain frozen).

Action code supervision signal:
* Emerges solely from reconstruction + downstream dynamics consistency; no explicit behavior labels.

Why this order:
* Stable discrete visual vocabulary (tokenizer) is prerequisite so LAM & Dynamics learn over a fixed token space.
* Joint LAM + Dynamics encourages action codes to maximize future token predictability (functional semantics).

## 7. Results 

![Diverse trajectories](./figures/platformer_trajectories.png)
*Figure (Diverse trajectories):* Genie is a generative model that can be used as an interactive environment. The model can be prompted in various ways, either with a generated image (top) or a hand-drawn sketch (bottom). At each time step, the model takes a user-provided latent action to generate the next frame, producing trajectories with interesting and diverse character actions.

![Playing from Image Prompts](./figures/actions_emergent.png)
*Figure (Playing from Image Prompts):* We can prompt Genie with images generated by text-to-image models, hand-drawn sketches or real-world photos. In each case we show the prompt frame and a second frame after taking one of the latent actions four consecutive times. In each case we see clear character movement, despite some of the images being visually distinct from the dataset.

![Learning to simulate deformable objects](./figures/chips.png)
*Figure (Learning to simulate deformable objects):* we show frames from a ten step trajectory in the model, taking the same action. Genie is capable of learning the physical properties of objects such as bags of chips.

![Emulating parallax](./figures/parallax_new.png)
*Figure (Emulating parallax):* Emulating parallax, a common feature in platformer games. From this initial text-generated image, the foreground moves more than the near and far middle ground, while the background moves only slightly.

![Controllable, consistent latent actions in Robotics](./figures/action_grid_robotics.png)
*Figure (Controllable, consistent latent actions in Robotics):* trajectories beginning from three different starting frames from our Robotics dataset. Each column shows the resulting frame from taking the same latent action five times. Despite training without action labels, the same actions are consistent across varied prompt frames and have semantic meaning: down, up and left.

![Playing from RL environments](./figures/coinrun_traj.png)
*Figure (Playing from RL environments):* Genie can generate diverse trajectories given an image of an unseen RL environment.