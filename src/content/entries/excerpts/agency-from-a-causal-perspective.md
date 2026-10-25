---
title: "Agency from a Causal Perspective"
date: "2024-10-27"
type: "excerpt"
subtype: "article-summary"
topic: "Alignment, safety & AGI"
source_title: "Agency from a Causal Perspective"
source_author: null
source_url: null
---

### Agency and its Importance

- **Agency:** In the context of this source, agency is interpreted as **goal-directedness**, meaning a system acts as if it is trying to steer the world in a particular direction. This interpretation is crucial as it sets the stage for analyzing AI systems based on their goal-oriented behaviour.
- **Importance of Understanding Agency:** A deep understanding of agency in AI systems is vital for several reasons:
    - Assessing the potential dangers and harms of powerful AI systems.
    - Evaluating whether an AI model possesses a concerning level of agency.
    - Designing systems that are either not agentic (e.g., AI scientists, oracles) or exhibit agency in a safe and controlled manner.
    - Establishing a foundation for advancements in other crucial AGI safety areas such as interpretability, incentives, and generalisation.
    - Preserving human agency, by understanding the factors that enhance or diminish it.

### Key Dimensions of Agency

The source presents eight key dimensions of agency, each offering insights into the nature and extent of an AI system's goal-directed behaviour:

- **Degrees of Freedom:** An agent needs the ability to choose among different options. This choice doesn't have to be free from causal influences; the essence is the potential for different outcomes. For instance, a thermostat has limited degrees of freedom (controlling only the heater output), while a human has a much wider range of physical and verbal actions.
- **Influence:** An agent's actions must have consequences in the world. The extent of this influence can vary greatly. A language model's influence depends on its reach – whether it interacts only with developers or with millions of users.
- **Adaptation:** This is the most crucial dimension. Goal-directed agents act based on preferences, and these preferences shape their actions. Adaptation involves a feedback loop where consequences influence future behaviour. Mr. Jones watering his lawn is an example. He turns on the sprinkler _because_ it makes the grass green. If the grass didn't need water, he wouldn't water it.
- **Coherence:** This refers to how consistently an agent pursues a long-term goal. Governments, for example, can maintain focus on infrastructure projects for decades, whereas current language models tend to deviate from their goals quickly.
- **Self-Preservation:** An agent that doesn't persist can't pursue a goal. Humans exhibit a strong drive for self-preservation. Current language models show some degree of self-preservation, while systems like GPS navigation show none.
- **(Markov) Separation from the Environment:** This refers to the agent's ability to maintain a degree of separation from its environment, often through boundaries like a cell wall, skin, or encryption.
- **Information or Perception:** The amount of information an agent has about its environment significantly impacts its ability to act effectively.

### Visualizing Agency: Mechanised Causal Graphs

The source employs mechanised causal graphs to represent the concepts of influence and adaptation. These graphs use:

- **Object-level Nodes:** Representing real-world entities or events, such as the sprinkler or the grass in the example.
- **Mechanism Nodes:** Representing the causal mechanisms that govern how object-level nodes behave. For example, Mr. Jones’ watering policy is represented by a mechanism node linked to the sprinkler.

These graphs help visualise how an agent's actions influence the environment and how feedback from the environment affects the agent's future decisions, highlighting the dynamic interplay between an agent and its surroundings.

### Detecting and Measuring Agency

The source suggests that **adaptation can be detected using causal discovery algorithms**, opening up possibilities for identifying agents within complex systems. It also proposes quantitative measures for various dimensions of agency:

- **Adaptation:** Speed, effectiveness, robustness, and retargetability can quantify adaptation, providing insights into how well an agent adjusts to changes.
- **Coherence:** Comparing goals across different decision nodes can assess coherence, helping to understand the consistency of an agent's pursuits.
- **Influence:** Measures like causal channel capacity and performative power can be used to quantify an agent's influence on its environment.

These measurements can contribute to evaluating the capabilities and potential risks associated with AI systems.

### Far-Reaching Implications

The concepts presented in the source have significant implications for the future of AI development, safety, and our understanding of agency itself:

- **Designing Safer AI:** A deeper understanding of agency can guide the design of AI systems that are inherently safer by controlling or limiting potentially harmful dimensions of agency, such as self-preservation or the pursuit of misaligned goals.
- **Evaluating AI Risks:** By measuring different dimensions of agency, we can better assess the potential dangers posed by AI systems. This can inform decisions about deployment and regulation.
- **Understanding the Emergence of Agency:** The framework presented can be used to investigate how agency emerges from simpler components, shedding light on the origins of complex behaviour in both natural and artificial systems.
- **Preserving Human Agency:** By understanding how technology impacts human agency, we can design systems that enhance human capabilities rather than undermine our autonomy.

The insights offered by this source contribute to a more nuanced understanding of agency in AI, paving the way for the development of safer, more beneficial AI systems while addressing the ethical and societal challenges they present.
