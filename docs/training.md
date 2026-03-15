### Architectural Impact on Convergence

| ⚙️ Core Optimization & Routing | 🧠 Attention Mechanisms |
| :---: | :---: |
| ![Optimizer Convergence](assets/loss_by_optimizer.png){ width="100%" } | ![MoE Impact](assets/loss_by_moe.png){ width="100%" } |
| **Convergence by Optimizer:** Isolating the impact of the optimization algorithm across identical architectures. | **Sparse MoE vs Dense:** Evaluating the convergence speed when routing parameters through Top-K experts. |
| ![Sliding Window](assets/loss_by_sliding_window.png){ width="100%" } | ![Attention Sink](assets/loss_by_no_sink.png){ width="100%" } |
| **Sliding Window:** Impact of restricting the attention receptive field on the learning trajectory. | **Attention Sink Gating:** Training stability achieved by applying a sigmoid gate (`no_sink`) to attention outputs. |

---

### Memory & Parameter Efficiency

| 🔗 Parameter Sharing | 💾 Memory Footprint |
| :---: | :---: |
| ![Weight Tying](assets/loss_by_weight_tying.png){ width="100%" } | ![VRAM Footprint](assets/vram_comparison.png){ width="100%" } |
| **Weight Tying:** Convergence behavior when tying the embedding matrix to the output language modeling head. | **Peak VRAM (Dense vs Sparse MoE):** Scaling capacity via MoE while maintaining a constrained VRAM footprint. |