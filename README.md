# HulluEdit Extensions

Course project for the Fall 2026 seminar [Trustworthy AI Models and Systems](https://github.com/lydiaYchen/26TrustAI). This project explores three extensions to [HulluEdit](https://arxiv.org/abs/2602.22727), a method for mitigating object hallucinations in vision-language models.

| Extension | Status |
| --- | --- |
| [PDE](PDE/) · Preference-Constrained Direction Editing | LLaVA-1.5-7B inference and CHAIR evaluation |
| [VMC](VMC/) · Visual Memory Calibration | LLaVA-1.5-7B adapter training, inference, and evaluation |
| [ST](ST/) · Audio Hallucination Mitigation | Audio HulluEdit and PDE on Qwen audio-language models |

PDE refines HulluEdit's update within the prior subspace at each beam-search step. It constrains the size of the update and how much of HulluEdit's preference signal to retain. See the [PDE setup and run instructions](PDE/README.md) to generate captions and evaluate them on COCO 2014.

This project builds on [Lin et al., CVPR 2026](https://arxiv.org/abs/2602.22727) and their [HulluEdit implementation](https://github.com/VioAgnes/HulluEdit).
