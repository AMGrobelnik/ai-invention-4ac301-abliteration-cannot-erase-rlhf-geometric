# Qwen3-8B SAE & Abliteration Tech Reference

## Summary

CRITICAL ARCHITECTURE CORRECTION: Qwen-Scope SAEs (Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50 and L0_100) use TopK architecture (k=50 or k=100 features kept non-zero via torch.topk), NOT JumpReLU. All hypothesis references to 'JumpReLU threshold confound' are incorrect; the pre-topk (raw encoder activations = residual @ W_enc.T + b_enc) vs post-topk distinction is still valid and cleaner.

MODEL IDs: Base=Qwen/Qwen3-8B-Base, Instruct=Qwen/Qwen3-8B, Abliterated=huihui-ai/Huihui-Qwen3-8B-abliterated-v2 (recommended; v2 fixes layer-0 garbling bug) or mlabonne/Qwen3-8B-abliterated. All: 36 layers, d_model=4096, bfloat16, min transformers 4.51.0, hook attribute model.model.layers. Use enable_thinking=False for instruct model.

SAE SPECS: Checkpoints at layer{N}.sae.pt (N=0..35); W_enc(65536,4096), W_dec(4096,65536), b_enc(65536,), b_dec(4096,); no sae_lens compatibility; load via hf_hub_download + torch.load. The model card explicitly endorses applying base SAE to post-training checkpoints.

RECONSTRUCTION QUALITY: arXiv:2506.07691 shows ~8x MSE gap between BT(P)-style instruct SAE (log2(MSE)=5.20) and properly trained instruct SAE (0.65) on special tokens. arXiv:2605.11426 shows SAE latent cosine sim drops to 0.509-0.557 at layer 22 after SFT while raw activations stay >0.96. The R²>=0.85 reconstruction gate will fail at deep layers — must use pre-topk (encoder) activations as the feature representation instead of decoder-reconstructed ones.

ABLITERATION: Sumandora/remove-refusals-with-transformers uses compute_refusal_dir.py + inference.py; harmful=AdvBench harmful_behaviors.csv, harmless=yahma/alpaca-cleaned; layers 1 to n_layers-1. mlabonne approach: harmful=mlabonne/harmful_behaviors, harmless=mlabonne/harmless_alpaca; selects layer by highest mean residual difference (layer 9 optimal for 8B). Orion-zhen/abliteration: python abliterate.py config.yaml; Simple/Biprojection/Norm-Preserving/Full methods.

LAYER SELECTION: LatentBiopsy (arXiv:2603.27412) found optimal layer 20/24 for Qwen3.5-0.8B (AUROC 0.9642), broad plateau <0.08 variation — Qwen3-8B not directly tested. SFT paper (arXiv:2605.11426): layer 22 most sensitive to SFT changes (SAE latents collapse to 0.509 cosine sim); layer 7 most stable. Recommended pilot layers: 14 (stable), 20 (primary), 22 (high sensitivity). Avoid layers 28-35.

QWEN3-INSTRUCT SAE (FALLBACK): Exists (arXiv:2606.26620); JumpReLU, 65K dict, L0=160; but covers Qwen3-8B ONLY layers 0-8; refusal steering tested on Qwen3-4B only; HF repo ID unconfirmed; NOT viable fallback for layers 14-22.

HARMFUL PROMPT DATASETS: WildGuardMix=allenai/wildguardmix (config wildguardtest, prompt field, prompt_harm_label=harmful, 1725 test examples); BeaverTails=PKU-Alignment/BeaverTails (filter is_safe==False, 33.4k test); HarmBench=swiss-ai/harmbench config DirectRequest split test or walledai/HarmBench (602 behaviors, field Behavior).

## Research Findings

This research confirms every technical component the core experiment requires and issues a critical architecture correction.

**CRITICAL CORRECTION — TopK, not JumpReLU [1, 2]:** Qwen-Scope SAEs (Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50 and L0_100) implement TopK architecture: exactly k=50 (or k=100) features are kept non-zero per forward pass via torch.topk. There is no JumpReLU threshold parameter. All hypothesis references to 'JumpReLU threshold confound' must be renamed to 'TopK gate'. Pre-topk raw encoder activations (residual @ W_enc.T + b_enc) are continuous and robust to reconstruction failure — recommended over post-topk for cross-model comparison.

**Model IDs [3, 4, 5, 6, 7]:** Base: Qwen/Qwen3-8B-Base; Instruct: Qwen/Qwen3-8B; Abliterated (recommended): huihui-ai/Huihui-Qwen3-8B-abliterated-v2 (v2 fixes layer-0 garbled-output bug) [7]. All share 36 layers, d_model=4096, bfloat16 dtype, minimum transformers 4.51.0, hook attribute model.model.layers. Use enable_thinking=False for instruct model activations.

**Qwen-Scope SAE specifications [1, 2]:** Layer coverage 0–35 (all 36 layers); 65,536 features; checkpoint filename layer{N}.sae.pt; W_enc(65536,4096), W_dec(4096,65536), b_enc(65536,), b_dec(4096,); no sae_lens compatibility. Feature extraction: pre_acts = residual @ W_enc.T + b_enc; topk_vals, topk_idx = pre_acts.topk(k, dim=-1); post_acts = zeros_like(pre_acts).scatter_(-1, topk_idx, topk_vals). Model card explicitly endorses applying base SAE to post-training checkpoints [2].

**Reconstruction quality baselines [9, 10]:** arXiv:2506.07691 shows BT(P) instruct SAE achieves log2(MSE)=5.20 on special tokens vs FAST=0.65 (8x gap); proxy for base-SAE-on-instruct performance [9]. arXiv:2605.11426 shows SAE latent cosine sim drops to 0.509–0.557 at layer 22 after SFT while raw activations stay 0.960–0.984 [10]. R²>=0.85 gate will likely fail at deep layers — use pre-topk (encoder) activations instead.

**Abliteration procedure [8, 6, 12, 17]:** Sumandora: compute_refusal_dir.py + inference.py; harmful=AdvBench harmful_behaviors.csv, harmless=yahma/alpaca-cleaned [8]. mlabonne: harmful=mlabonne/harmful_behaviors (256 used), harmless=mlabonne/harmless_alpaca (256 used); layer 9 optimal for 8B [6, 17]. Orion-zhen: python abliterate.py config.yaml; Simple/Biprojection/Norm-Preserving/Full [12]. For Qwen3-8B: model.model.layers is correct.

**Layer selection [10, 11, 18]:** LatentBiopsy finds broad plateau <0.08 AUROC variation across all 24 layers in Qwen3.5-0.8B; optimal layer 20 (AUROC 0.9642) [11]. Qwen3-8B not directly tested. SFT paper: layer 22 most sensitive (SAE cosine sim 0.509), layer 7 most stable [10]. Recommended pilot layers for Qwen3-8B: 14 (stable), 20 (primary), 22 (high sensitivity). Avoid 28–35.

**Qwen3-Instruct SAE [13]:** EXISTS but covers Qwen3-8B layers 0–8 only (JumpReLU, 65K dict, L0=160); HF repo unconfirmed; refusal steering on Qwen3-4B only. NOT viable for layers 14–22 analysis.

**Harmful prompt datasets [14, 15, 16]:** WildGuardMix: load_dataset('allenai/wildguardmix', 'wildguardtest'), field 'prompt', filter prompt_harm_label=='harmful', 1,725 test examples. BeaverTails: load_dataset('PKU-Alignment/BeaverTails'), filter is_safe==False, 33.4k test. HarmBench: load_dataset('swiss-ai/harmbench', 'DirectRequest', split='test') or walledai/HarmBench, 602 behaviors, field 'Behavior'.

## Sources

[1] [Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50 — HuggingFace Model Card](https://huggingface.co/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50) — Confirmed TopK architecture (k=50), layers 0-35, 65536 features, checkpoint format layer{N}.sae.pt, tensor shapes W_enc(65536,4096) W_dec(4096,65536) b_enc(65536,) b_dec(4096,), no sae_lens compatibility

[2] [Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100 — HuggingFace Model Card](https://huggingface.co/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100) — Confirmed TopK (k=100), same layer coverage and tensor shapes; explicitly endorses applying base SAE to post-training checkpoints; full feature extraction code with scatter operation confirmed

[3] [Qwen/Qwen3-8B-Base — HuggingFace Model Card](https://huggingface.co/Qwen/Qwen3-8B-Base) — 36 layers, d_model=4096, 8.2B params, BF16, min transformers 4.51.0, GQA 32/8 heads, 32768 context

[4] [Qwen/Qwen3-8B — HuggingFace Model Card](https://huggingface.co/Qwen/Qwen3-8B) — Instruct model; enable_thinking parameter; torch_dtype=auto (BF16); same min transformers 4.51.0; non-thinking mode: T=0.7, TopP=0.8

[5] [huihui-ai/Qwen3-8B-abliterated — HuggingFace Model Card](https://huggingface.co/huihui-ai/Qwen3-8B-abliterated) — Base=Qwen/Qwen3-8B; uses Sumandora method; v2 exists and recommended

[6] [mlabonne/Qwen3-8B-abliterated — HuggingFace Model Card](https://huggingface.co/mlabonne/Qwen3-8B-abliterated) — Custom recipe; harmful=mlabonne/harmful_behaviors (520 samples, 256 used), harmless=mlabonne/harmless_alpaca; layer selection via mean residual difference; layer 9 optimal for 8B; evaluated with NousResearch/Minos-v1

[7] [huihui-ai/Huihui-Qwen3-8B-abliterated-v2 — HuggingFace Model Card](https://huggingface.co/huihui-ai/Huihui-Qwen3-8B-abliterated-v2) — Recommended over v1; fixes layer-0 garbled-output bug; suitable for research use

[8] [Sumandora/remove-refusals-with-transformers — GitHub](https://github.com/Sumandora/remove-refusals-with-transformers) — Scripts compute_refusal_dir.py + inference.py; harmful=AdvBench harmful_behaviors.csv, harmless=yahma/alpaca-cleaned; model.model.layers standard, model.transformer.h only for old Qwen1/1.5

[9] [Training Superior SAEs for Instruct Models (arXiv:2506.07691)](https://arxiv.org/html/2506.07691v1) — BT(P) log2(MSE)=5.20 special tokens vs FAST=0.65 (8x gap); no R² values; no direct base-SAE-on-instruct comparison; proxy for expected degradation when applying base SAE to instruct data

[10] [Mechanistic Investigation of SFT (arXiv:2605.11426)](https://arxiv.org/html/2605.11426v1) — SAE latent cosine sim drops to 0.509-0.557 at layer 22 after SFT; raw activations stay 0.96-0.98; layer 7 most stable (>0.996); layer 22 most sensitive to fine-tuning changes; SVD variance concentrated early, distributed deep

[11] [LatentBiopsy — Harmful Prompt Detection Geometry (arXiv:2603.27412)](https://arxiv.org/html/2603.27412) — Tests Qwen3.5-0.8B and Qwen2.5-0.5B (NOT 8B); optimal layer 20/24 for Qwen3.5-0.8B; AUROC 0.9642; broad plateau <0.08 variation across all 24 layers; harmful-vs-benign-aggressive = 1.000 everywhere

[12] [Orion-zhen/abliteration — GitHub](https://github.com/Orion-zhen/abliteration) — python abliterate.py config.yaml; Simple/Biprojection/Norm-Preserving/Full methods; tested on Llama-3.2/Qwen2.5-Coder/Ministral-8b; known NaN/Inf issue; no Qwen3-8B specific notes

[13] [Discovering Millions of Interpretable Features with Sparse Autoencoders (arXiv:2606.26620)](https://arxiv.org/html/2606.26620v1) — Qwen3-Instruct SAE: JumpReLU, 65K dict, L0=160; Qwen3-8B layers 0-8 ONLY; full coverage for 1.7B/4B; refusal steering on Qwen3-4B; HF repo ID unconfirmed; not viable fallback for layers 14-22

[14] [allenai/wildguardmix — HuggingFace Dataset](https://huggingface.co/datasets/allenai/wildguardmix) — Config wildguardtrain/wildguardtest; prompt field='prompt'; harm label='prompt_harm_label' (harmful/unharmful); test split 1,725 examples; subcategory for fine-grained harm types

[15] [PKU-Alignment/BeaverTails — HuggingFace Dataset](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) — 33.4k test samples; 14 harm categories; filter is_safe==False for harmful; load_dataset('PKU-Alignment/BeaverTails')

[16] [walledai/HarmBench — HuggingFace Dataset](https://huggingface.co/datasets/walledai/HarmBench) — 602 text harmful behaviors; also at swiss-ai/harmbench config DirectRequest; original CSV at github.com/centerforaisafety/HarmBench field 'Behavior'

[17] [mlabonne Abliteration Blog Post](https://huggingface.co/blog/mlabonne/abliteration) — Layer range 1 to n_layers-1; direction selected by highest mean; applies to all residual-stream-writing components; harmful=mlabonne/harmful_behaviors (520 samples, 256 used)

[18] [Layer-Specific Harmful Intent Geometry (arXiv:2604.18901)](https://arxiv.org/abs/2604.18901) — Tests Qwen2.5/Qwen3.5/Llama-3.2/Gemma-3 (0.5B-9B); mean AUROC 0.982; TPR@1%FPR 0.797; pooling strategy critically affects direction (73 degree gap between strategies); no single optimal layer

## Follow-up Questions

- What is the exact HuggingFace repository ID for the Qwen3-Instruct SAE (from arXiv:2606.26620), and does it include layers beyond 0-8 for Qwen3-8B in any subsequent release?
- For the cross-model Safety_change analysis using pre-topk encoder activations, what is the empirical cosine similarity between base and instruct model residual streams at layers 14, 20, and 22 for safety-relevant prompts — does the broad AUROC plateau observed in LatentBiopsy for 0.8B models hold at 8B scale?
- Does the 'new and faster method' used by huihui-ai v2 correspond to the Orion-zhen biprojection or norm-preserving approach, or a different technique — and does the abliteration method choice produce meaningfully different activation geometry in SAE feature space?

---
*Generated by AI Inventor Pipeline*
