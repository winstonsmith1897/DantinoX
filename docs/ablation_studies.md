# Ablation Studies

To rigorously validate the architectural choices within DantinoX, we conducted extensive hyperparameter sweeps using **Weights & Biases (W&B)**. Instead of relying on conventional wisdom, every major component—from routing penalties to attention mechanics—was empirically tested against hardware constraints and convergence stability.

The insights below are derived from analyzing the joint distribution of validation loss (`val_loss`), peak memory footprint (`vram_gb`), and execution speed (`ms_per_step`) across hundreds of Bayesian trials.

---

## 1. Model Capacity & Convergence Dynamics

This section analyzes how fundamental model scaling laws (depth vs. width) and core training hyperparameters impact the final language modeling performance.

### Depth, Width, and Batch Scaling

| 🏗️ Average Loss Heatmap | 📦 Effective Batch Size |
| :---: | :---: |
| ![Capacity Loss Heatmap](assets/wandb_insights/insight_capacity_heatmap.png){ width="100%" } | ![Effective Batch Size](assets/wandb_insights/insight_effective_batch_size.png){ width="100%" } |
| **Average Loss by Architecture:** A contour map illustrating the sweet spot between model depth (`num_blocks`) and width (`dim`). It answers the critical scaling question: is it better to add layers or increase the hidden dimension? | **Validation Loss vs Batch Size:** Boxplot distribution demonstrating the variance and stabilization of the validation loss as the global effective batch size increases. |

### Optimizer & Learning Rate Sensitivity

| Learning Rate vs Optimizer |
| :---: |
| ![LR vs Optimizer](assets/wandb_insights/insight_lr_vs_optimizer.png){ width="80%" } |
| **Optimizer Convergence Basins:** A logarithmic analysis comparing AdamW, Adafactor, and Lion across different learning rates, identifying the most stable convergence basin for each. |

---

## 2. Architecture Specifics & Memory Efficiency

Modern LLM engineering is primarily memory-bound. This section proves the efficiency of the advanced architectural features implemented in DantinoX to reduce the GPU footprint.

### Parameter Sharing & Attention Optimizations

| Weight Tying VRAM Savings |
| :---: |
| ![Weight Tying VRAM](assets/wandb_insights/perf_weight_tying_vram.png){ width="80%" } |
| **Weight Tying:** Empirical verification of VRAM reduction achieved by sharing the embedding matrix with the output LM head across different model dimensions. |

### VRAM Scaling Laws

| Context Length Memory Cost | Model Capacity VRAM Heatmap |
| :---: | :---: |
| ![VRAM vs Context](assets/wandb_insights/perf_vram_vs_context.png){ width="100%" } | ![VRAM Capacity Heatmap](assets/wandb_insights/perf_vram_capacity_heatmap.png){ width="100%" } |
| **Context Window:** Proves the linear/quadratic VRAM cost associated with expanding the sequence length, highlighting the critical need for optimizations like Sliding Window attention. | **VRAM Lookup Heatmap:** A visual reference guide mapping hidden dimension (`dim`) and layer count (`num_blocks`) against peak VRAM usage. This allows for instant hardware requirement estimation. |

---

## 3. Sparse Mixture of Experts (MoE) Analysis

Implementing MoE in JAX requires careful balancing of speed overhead and routing quality. This section provides a deep dive into the performance trade-offs of the gated MLP blocks.

| Dense vs MoE Step Time | Balancing Penalty Trade-off |
| :---: | :---: |
| ![MoE Step Time Overhead](assets/wandb_insights/perf_moe_step_time.png){ width="100%" } | ![MoE Alpha Penalty](assets/wandb_insights/insight_moe_alpha_penalty.png){ width="100%" } |
| **Routing Overhead:** Direct comparison of milliseconds-per-step between Dense and MoE models, quantifying the actual XLA compilation and execution cost of token routing. | **Expert Load Balancing:** Regression plot illustrating the tension between routing fairness (Alpha Balance penalty) and cross-entropy loss. High penalties force expert usage but may degrade final language modeling accuracy. |

---

## 4. Regularization & Training Efficiency

The final section evaluates how to control overfitting and maximize the utilization of hardware resources during the training loop.

| Dropout Effectiveness by Model Size | Gradient Accumulation Overhead |
| :---: | :---: |
| ![Dropout vs Capacity](assets/wandb_insights/insight_dropout_vs_capacity.png){ width="100%" } | ![Grad Accum Speed](assets/wandb_insights/perf_grad_accum_speed.png){ width="100%" } |
| **Dropout Regularization:** Analyzes the interaction between dropout rate and model capacity. It proves the rule of thumb that smaller models degrade with high dropout, while larger models require it to prevent overfitting. | **Execution Speed:** Boxplot distribution showing how execution time scales with gradient accumulation steps. While accumulation saves VRAM, it introduces a subtle time penalty per virtual step. |


---

## Appendix: Complete Parameter Distributions

For full transparency and reproducibility, the following expandable section contains the isolated distributions of every hyperparameter swept during the Bayesian optimization process, plotted against the target validation loss. 

??? abstract "Click to expand all Base Distributions (Boxplots & Scatter Plots)"

    ### Categorical Architectural Choices (Boxplots)
    These plots demonstrate the variance and median validation loss across boolean toggles and categorical selections.

    | Core & Routing | Attention & Positional |
    | :---: | :---: |
    | ![Optimizer](assets/wandb_insights/base_box_optimizer.png){ width="100%" } | ![Attention Sink](assets/wandb_insights/base_box_no_sink.png){ width="100%" } |
    | ![MoE Toggle](assets/wandb_insights/base_box_use_moe.png){ width="100%" } | ![Sliding Window](assets/wandb_insights/base_box_sliding_window.png){ width="100%" } |
    | ![SwiGLU Toggle](assets/wandb_insights/base_box_use_swiglu.png){ width="100%" } | ![Positional Encoding](assets/wandb_insights/base_box_pos_encoding.png){ width="100%" } |
    | ![Weight Tying](assets/wandb_insights/base_box_weight_tying.png){ width="100%" } | ![Tokenizer Type](assets/wandb_insights/base_box_tokenizer_type.png){ width="100%" } |

    ---

    ### Numeric Hyperparameters (Scatter Plots)
    These plots isolate continuous and discrete numerical values, complete with Spearman correlation trends.

    | Training Dynamics | Memory & Context |
    | :---: | :---: |
    | ![Learning Rate](assets/wandb_insights/base_scatter_lr.png){ width="100%" } | ![Max Context](assets/wandb_insights/base_scatter_max_context.png){ width="100%" } |
    | ![Batch Size](assets/wandb_insights/base_scatter_batch_size.png){ width="100%" } | ![Context Window (SW)](assets/wandb_insights/base_scatter_context_window.png){ width="100%" } |
    | ![Effective Batch Size](assets/wandb_insights/base_scatter_effective_batch_size.png){ width="100%" } | ![Gradient Accumulation](assets/wandb_insights/base_scatter_grad_accum.png){ width="100%" } |
    | ![Warmup Steps](assets/wandb_insights/base_scatter_warmup_steps.png){ width="100%" } | ![Dropout Rate](assets/wandb_insights/base_scatter_dropout_rate.png){ width="100%" } |

    | Architecture Dimensions | Mixture of Experts (MoE) |
    | :---: | :---: |
    | ![Hidden Dimension](assets/wandb_insights/base_scatter_dim.png){ width="100%" } | ![Number of Experts](assets/wandb_insights/base_scatter_n_experts.png){ width="100%" } |
    | ![Number of Blocks](assets/wandb_insights/base_scatter_num_blocks.png){ width="100%" } | ![Top-K MLP](assets/wandb_insights/base_scatter_top_k_mlp.png){ width="100%" } |
    | ![KV Heads (GQA)](assets/wandb_insights/base_scatter_kv_heads.png){ width="100%" } | ![Alpha Balance](assets/wandb_insights/base_scatter_alpha_balance.png){ width="100%" } |
    | ![Expansion Factor](assets/wandb_insights/base_scatter_expansion.png){ width="100%" } | |