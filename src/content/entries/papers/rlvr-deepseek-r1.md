---
title: "RLVR - DeepSeek R1"
date: "2025-05-16"
type: "paper-summary"
subtype: null
topic: "RL for LLMs"
source_title: "DeepSeek-R1"
source_author: "DeepSeek"
source_url: null
---

## Opening (3 minutes)
* This is a "mainstream" paper, it has key innovations but also engages in more of "hype building" as well as a bit of speculation and "LLM psychology"
-  "What if we could teach AI to reason without showing it a single example?"
- This paper shows you can get SOTA reasoning with pure RL - no supervised examples needed initially
- Two main models: R1-Zero (pure RL) and R1 (multi-stage with some examples)
- **Key achievement**: Matches OpenAI o1-1217 on most benchmarks
## Main Innovation - DeepSeek-R1-Zero (4 minutes)

_Reference pages 3-9_
- **Big deal**: First to prove reasoning can emerge from pure RL
- Started with just:
    - Base model (DeepSeek-V3)
    - Simple template: `<think>reasoning</think><answer>final answer</answer>`
    - Rule-based rewards (only correct/incorrect)
- **Show Figure 2** (page 7): AIME accuracy going from 15.6% to 71.0%
- **Show Figure 3** (page 8): Response length naturally growing longer
- **Read the "Aha Moment" example** (page 9, Table 3):
    - "Wait, wait. Wait. That's an aha moment..."
    - Model develops self-awareness and reflection
    - Uses human-like language naturally
    - This wasn't programmed!

## The Multi-Stage Pipeline (5 minutes)

_Reference pages 9-11_

### Stage 1: Cold Start (Section 2.3.1)

- Problem: Pure RL gives poor readability and language mixing
- Solution: Use thousands of long CoT examples to set format
- Creates readable pattern: reasoning process | summary

### Stage 2: Reasoning-Oriented RL (Section 2.3.2)

- Same as R1-Zero but starting from Stage 1 checkpoint
- Add language consistency reward (slight performance drop but better readability)

### Stage 3: Rejection Sampling + SFT (Section 2.3.3)

- Take RL checkpoint, generate lots of responses
- Keep only correct ones (600k reasoning samples)
- Add 200k non-reasoning data (writing, QA)
- **Key insight**: Start fresh from base model with this new dataset

### Stage 4: Full-Scenario RL (Section 2.3.4)

- Apply RL to all scenarios (not just reasoning)
- Add human preference alignment
- Final DeepSeek-R1 model

## Results Deep Dive (3 minutes)

_Reference Table 4 (page 13)_

- **Mathematics**: 79.8% on AIME (beats o1-1217's 79.2%)
- **Reasoning**: 97.3% on MATH-500 (matches o1-1217's 96.4%)
- **Coding**: 96.3% percentile on Codeforces (matches o1)
- **General**: Strong on MMLU, Arena-Hard
- **Key point**: This is open research matching closed models!

## Distillation Results (2 minutes)

_Reference Table 5 (page 14)_

- **Major finding**: Distillation > Direct RL on small models
- DeepSeek-R1-Distill-Qwen-7B: 55.5% on AIME (beats QwQ-32B-Preview!)
- 14B model: 69.7% on AIME
- 32B model: 72.6% on AIME (comparable to o1-mini)
- **Insight**: Teacher model size matters more than student RL compute

## What Didn't Work (2 minutes)

_Reference Section 4.2 (pages 15-16)_

1. **Process Reward Models**:
    - Suffer from reward hacking
    - Hard to define fine-grained steps
    - Add complexity without big gains
2. **Monte Carlo Tree Search**:
    - Works for inference, not iterative improvement
    - Search space too large compared to chess/Go
    - Value model too hard to train
3. **Pure RL limitations**:
    - Poor readability
    - Language mixing issues
    - Needs careful prompt engineering

## Key Insights & Philosophy (2 minutes)

### Technical Insights:

- **Emergence is real**: Complex behaviors from simple rewards
- **Scale matters**: Larger models benefit more from RL
- **Multi-stage works**: Iterate between RL and SFT
- **Distillation is powerful**: Better than training small models directly

### Philosophical Questions:

- **Is long reasoning a bug or feature?** Like humans need thinking time
- **What does emergence tell us?** Intelligence might naturally develop from optimization
- **Safety implications**: Hard to predict what behaviors will emerge

## Questions for me 

1. "If reasoning is emergent, what does this tell us about the nature of intelligence?"
2. "Should we view the need for 'thinking time' as more human-like or as a limitation?"
3. "What are the implications for AI safety if sophisticated behaviors emerge unexpectedly?"

## Quick Reference Points to Emphasize:

- **"First to show pure RL can develop reasoning"**
- **"The aha moment shows genuine emergence"**
- **"Matches OpenAI o1 performance with open methods"**
- **"Distillation works better than expected"**
- **"Many sophisticated behaviors emerged naturally"**
