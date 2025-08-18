# About Genie: Generative Interactive Environments

A concise, practical guide to the Genie paper ("Genie: Generative Interactive Environments") – focusing on architecture, training workflow, reproducibility strategy, and how to reuse the core ideas.

Source paper LaTeX in this repo: `papers/genie-tex/`. Project website (examples & demos): https://sites.google.com/view/genie-2024/home

---
## 1. High-Level Idea
Genie is a *foundation world model* trained **only on unlabeled Internet videos** (no actions, no text) that becomes *interactive* by learning a **latent discrete action space**. A user (or agent) can press one of a small number of latent "buttons" at each timestep to advance the generated world frame-by-frame.

Core pillars:
- Unsupervised discovery of controllable latent actions (VQ codebook, tiny discrete set, e.g. 8 codes).
- Spatiotemporal (ST) Transformer everywhere for efficiency (separate spatial + temporal attention blocks).
- MaskGIT-style iterative token prediction for next-frame video generation.
- Two-stage pipeline: (1) learn video tokenizer (VQ-VAE over space+time) (2) co-train latent action model + dynamics model.

---
## 2. Main Components
| Component | Purpose | Key Tricks |
|-----------|---------|------------|
| Video Tokenizer (ST-ViViT VQ-VAE) | Compress frames into discrete tokens `z_t` | Temporal-aware encoding; linear-ish scaling in frames |
| Latent Action Model (LAM) | Infer discrete latent actions `a_t` between frames | Pixel input (not tokens) improves controllability; VQ codebook size ≤10 |
| Dynamics Model (ST-MaskGIT) | Predict next frame tokens conditioned on past tokens + latent actions | Additive action embeddings; random masking schedule |

---
## 3. Training Workflow (End-to-End)
1. Collect raw videos (e.g. 2D platformer gameplay clips; filtered & preprocessed to uniform FPS/resolution).
2. Train **Video Tokenizer** (VQ-VAE with ST-transformer encoder/decoder) on sequences of length T (e.g. 16 frames @10 FPS).
3. Freeze tokenizer; produce token sequences `z_{1:T}` for training dynamics.
4. Jointly train:
   - **Latent Action Model**: Given frames `x_{1:t}` and `x_{t+1}`, produce continuous latent, quantize to code `ã_t` (VQ). Decoder reconstructs `x_{t+1}`; only codebook retained for inference.
   - **Dynamics Model**: Given `(z_{1:t-1}, ã_{1:t-1})`, predict masked future tokens `z_t` (MaskGIT iterative refinement).
5. Inference: User supplies initial frame (image / sketch / generated image) + sequence of latent action indices; autoregressively generate new frames.

---
## 4. Latent Action Learning (Conceptual Algorithm)
Goal: Find a discrete code `ã_t` summarizing the *state change* from frame `t` to `t+1`.

Pseudo-code (conceptual):
```python
# Frames: x[0:T]
for t in range(T-1):
    h_t = action_encoder(x[0:t+1])          # ST-transformer w/ causal temporal mask
    a_cont_t = proj(h_t)                    # continuous latent
    a_code_t, commit_loss = vq_quantize(a_cont_t)  # nearest codebook vector
    x_pred_{t+1} = action_decoder(x[0:t+1], a_code_t)
    recon_loss += mse(x_pred_{t+1}, x[t+1])
loss = recon_loss + beta * commit_loss
```
Key design choices:
- **Pixel input beats token input**: preserves fine motion cues → higher controllability metric (ΔPSNR).
- **Small codebook (|A| ≈ 8)**: fosters stable semantics & playability.
- **Discard encoder/decoder at inference**: keep only the codebook; user selects code indices.

---
## 5. Video Tokenizer (ST-ViViT VQ-VAE)
Compresses each frame sequence into discrete tokens `z_t` where each token summarizes a spatial patch enriched with *causal past* temporal context (due to temporal attention ordering).

Benefits vs spatial-only:
- Temporal coherence baked into token representation.
- Fewer downstream steps for dynamics to infer motion.

Sketch:
```python
# Input: x shape (T, H, W, C)
patches = space_patchify(x, p)              # -> (T, N, p*p*C)
emb = linear(patches)
emb = add_positional_encoding(emb)
for blk in st_blocks:                       # each has spatial-attn then temporal-attn then FFN
    emb = blk(emb, causal_time=True)
quant_inputs = reshape_per_frame(emb)
z_t, vq_loss = vq(quant_inputs)             # discrete indices per patch
```

---
## 6. Dynamics Model (MaskGIT + ST Attention)
Predicts unknown (masked) tokens of future frames in parallel refinement steps.

Key points:
- Inputs: concatenated past frame tokens + additive embeddings for corresponding latent actions.
- Random masking rate U(0.5, 1.0) during training improves robustness.
- Iterative decoding per frame at inference (e.g. 25 refinement steps, temperature ~2).

Abstract loop:
```python
def predict_next(z_past, a_past):
    state = fuse(z_past, action_embed(a_past))
    for step in range(maskgit_steps):
        logits = st_transformer(state)
        masked_positions = sample_mask_positions(logits.confidence)
        state[masked_positions] = sample_tokens(logits[masked_positions], temp)
    return extract_new_frame_tokens(state)
```

---
## 7. Inference Loop (Putting It Together)
```python
# User-provided prompt image (can be sketch or generated artwork)
frame_1 = prompt_image
z_1 = tokenizer.encode(frame_1)
z_seq = [z_1]; a_seq = []

for t in range(1, T):
    a_index = user_choose_action(|A|)   # integer in [0, |A|)
    a_vec = action_codebook[a_index]
    a_seq.append(a_vec)
    z_next = dynamics.predict(z_seq, a_seq)
    z_seq.append(z_next)
    frame_next = tokenizer.decode(z_next)
    display(frame_next)
```
Interpretation learning: Users experiment with each latent button to map semantics (e.g. jump, move left, crouch). Semantics stay fairly consistent across prompts.

---
## 8. Metrics
- **FVD** (Frechet Video Distance): fidelity of generated sequences.
- **ΔPSNR (PSNRdiff)**: Controllability proxy:
  Δ = PSNR(real, generated_using_inferred_actions) − PSNR(real, generated_using_random_actions)
  Higher → actions matter.

---
## 9. Scaling Observations
- Larger dynamics model params (40M → 2.7B → 10B) monotonically improved training loss.
- Larger batch size similarly lowered loss (more tokens per update).
- Stability aids: bfloat16, QK-norm, staged component training.

Resource example (full-scale Genie): ~11B total params (tokenizer + LAM + dynamics), 16-frame context, large TPU pod scaling.

---
## 10. Practical Small-Scale Reproduction (Single GPU Prototype)
Approximate steps:
1. Collect ~50–200 hours of thematically consistent gameplay videos (downscale to 160×90 @ 10 FPS, clip length 16 frames).
2. Train mini video tokenizer:
   - Patch size 4 or 8, codebook size 512–1024, embedding dim 32–64.
   - 2–4 ST blocks (spatial attn over N patches, temporal attn causal over T).
3. Train latent action model:
   - Codebook size 6–8, patch size 16.
   - Use reconstruction + VQ losses; early stop when recon PSNR plateaus.
4. Train dynamics model:
   - Start small (e.g. 80–120M params, 8–12 ST blocks).
   - Random masking schedule; cross-entropy loss on token indices.
5. Evaluate ΔPSNR by swapping inferred vs random latent sequences.
6. Build a simple UI to send latent button presses and decode frames.

---
## 11. Applications & Extensions
- Rapid prototyping of interactive concept art (sketch → playable snippet).
- Data augmentation: generate synthetic trajectories for imitation or RL fine-tuning.
- Robotics: pre-train latent action abstractions from large, unlabeled corpora; later map to real actuators.
- Domain transfer: prompt with OOD imagery; adapt latent semantics via a small labeled mapping dataset.

---
## 12. Limitations / Open Challenges
| Challenge | Consequence | Possible Direction |
|-----------|-------------|--------------------|
| Short temporal window (e.g. 16 frames) | Limited long-horizon coherence | Memory-augmented transformers; hierarchical tokens |
| ~1 FPS inference | Not yet real-time | Parallel prediction; speculative decoding; distillation |
| Hallucinations / drift | Unrealistic trajectories over long rollouts | Consistency regularizers; latent planning layer |
| Fixed small action set | Coarse control | Hierarchical / compositional latent actions |
| No explicit physics grounding | Inaccurate dynamics in edge cases | Hybrid physical priors + learned tokens |

---
## 13. Adapting the Ideas
| Goal | Adaptation Tip |
|------|----------------|
| Add text conditioning | Concatenate text embedding to token sequence or prepend CLS-style tokens |
| Increase controllability | Enforce mutual information between actions and predicted change (InfoNCE) |
| More actions without confusion | Two-level codebook (coarse action → fine variant) |
| Long videos | Sliding window with state summary tokens |
| Faster decoding | Distill MaskGIT iterations into single-step policy (token-level teacher forcing) |

---
## 14. Reference Links (Primary & Cited Concepts)
- Genie Website: https://sites.google.com/view/genie-2024/home
- Transformers: Vaswani et al. 2017
- Vision Transformer (ViT): Dosovitskiy et al. 2021
- ST-Transformer (spatiotemporal attention idea): Xu et al. 2020
- VQ-VAE: Oord et al. 2017
- MaskGIT: Chang et al. 2022
- Phenaki (temporal-aware tokenization): Villegas et al. 2023
- RT-1 dataset (robotics video source): Brohan et al. 2023
- FVD metric: Unterthiner et al. 2019
- Behavioral cloning & world models (contextual works): listed throughout main paper

---
## 15. Quick Glossary
- **Latent Action**: Discrete code representing an inferred agent action between frames, learned unsupervised.
- **Token (Video)**: Discrete representation of a spatiotemporal patch produced by a VQ tokenizer.
- **ST Block**: Transformer block with separate spatial & temporal attention passes.
- **MaskGIT Steps**: Iterative refinement cycles sampling subsets of masked tokens each round.

---
## 16. Safety & Ethical Notes
- Generated content may inherit biases or IP-sensitive elements from Internet videos.
- Latent actions lack semantic guarantees; operator oversight needed before deployment (especially robotics).
- Releasing large, fully interactive generative environments raises moderation and misuse considerations.

---
## 17. Minimal Skeleton (Illustrative Only)
```python
class STBlock(nn.Module):
    def forward(self, x):  # x: (T, N, D)
        x = spatial_attn(x)          # per time step
        x = temporal_attn(x, causal=True)
        return ffw(x)

class VQCodebook:
    def quantize(self, z):
        # nearest neighbor in embedding space
        # return indices, embeddings, commitment loss
        ...

# Inference high-level
z = tokenizer.encode(prompt_frame)
state = [z]; actions = []
for step in range(num_steps):
    a = user_select_action()                  # int
    actions.append(codebook[a])               # embedding lookup
    z_next = dynamics.predict(state, actions)
    frame = tokenizer.decode(z_next)
    render(frame)
    state.append(z_next)
```
(Above is illustrative pseudocode; real implementation will handle batching, masking schedules, and parallel token refinement.)

---
## 18. Summary
Genie demonstrates that *interactive* latent-action world models can be trained from raw video alone by combining: (1) temporal-aware token compression, (2) unsupervised discrete action induction, and (3) iterative masked token prediction. This blueprint can be adapted to new domains by swapping datasets, adjusting codebook sizes, and extending context length.

---
*Prepared as a pragmatic hint/overview for quickly understanding and reusing the Genie architecture.*
