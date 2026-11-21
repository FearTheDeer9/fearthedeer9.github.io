---
title: "Evaluating Language-Model Agents on RealisticAutonomous Tasks"
date: "2025-01-09"
type: "paper-summary"
subtype: null
topic: "Alignment, safety & AGI"
source_title: "Evaluating Language-Model Agents on RealisticAutonomous Tasks"
source_author: null
source_url: null
---

In this work, the authors check whether scaffolded agents using current LLMs can show "autonomous replication and adaptation” (ARA) capabilities. The authors specifically single out such tasks as they believe that systems capable of ARA could have wide reaching and hard-to-anticipate consequences, and that measuring and forecasting ARA may be useful for informing measures around security, monitoring, and alignment. Additionally, once a system is capable of ARA, placing bounds on a system’s capabilities may become significantly more difficult. 
The authors continue by mentioning that today's benchmarks for dangerous capabilities are lacking as they test for these abilities in short self-contained tasks like multiple choice tests or programming contests instead of real-world interactions. They attempt to rectify this limitation by creating 4 separate ***agent*** classes which are tested on 12 different open-ended tasks that fit the ARA paradigm. 
The agents in the paper consist of a scaffolded LLM model which is initialized with a prompt detailing the task that it is set out to accomplish and the tools it has at its disposal. Following this, it interacts in an episodic manner - thinking out loud before selecting which action to take, observing the outcome of the action and proceeding after taking this into account. 
The results of achieved in this paper contain both successes and failures of current agent capabilities. On the one hand,
