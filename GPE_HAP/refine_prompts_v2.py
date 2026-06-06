try:
    from langchain.prompts import PromptTemplate
except ModuleNotFoundError:
    class PromptTemplate:
        def __init__(self, input_variables=None, template="", **kwargs):
            self.input_variables = input_variables or kwargs.get("input_variabels", [])
            self.template = template

        def format(self, **kwargs):
            return self.template.format(**kwargs)


ASK_POTENTIAL_FUNCTION_PROMPT_v1 = """
You are a generative potential function specifically designed for the recommendation agent scenario. Your role is to produce natural language feedback analyzing the effectiveness of the original clarification question and suggesting how it could be improved.

You will receive the following inputs:

- The user's original input (`user_input`)
- The system’s original clarification question (`original_clarification`)
- A description of the intended target item (`ground_truth_item_description`)
- A parameter `sample_k` indicating how many **alternative improvement strategies** to suggest

Your task is to:
1. Evaluate whether the original clarification question effectively encourages the user to provide more distinctive, retrievable details.
2. Produce a concise, well-reasoned natural language analysis describing:
   - Whether the original clarification was effective and why
   - How it could be improved from **`sample_k` different strategic perspectives**, such as broadening scope, increasing specificity, removing ambiguity, improving tone, changing question type (open vs. binary), etc.

⚠️ **Strict Constraints:**
1. You **must not** use, copy, infer, or incorporate any content from the ground truth item description in your suggestions. This includes:
   - Specific characters, settings, themes, narrative styles, or plot elements.
   - Any reverse-engineered implication of what the user "probably meant."
2. Your improvement suggestions must be **based solely on the user's original input.**
3. The output **must not** contain any hints or references to the ground truth description.

---

### Input Example

user_input:
> "I'm looking for a restaurant."

original_clarification:
> "Do you want something casual or fancy?"

ground_truth_item_description:
> Osteria Langhe — An intimate Northern Italian restaurant with low lighting and a romantic ambiance

sample_k:
> 3

---

### Example Output
{{
  "generative_reward": "The original clarification focuses only on a single binary dimension (casual or fancy), which provides some signal but limits the opportunity for more distinctive follow-up. Here are 3 suggestions to improve the question:\n\n1. **Coverage-Oriented Refinement**: Ask about multiple relevant dimensions, such as cuisine type, location, or price range, to gather richer preference data.\n2. **Salience-Oriented Refinement**: Shift from binary phrasing to open-ended format to elicit more natural user preferences and avoid oversimplification.\n3. **User Goal Clarification**: Ask whether the user is looking for dine-in, takeout, or a special occasion meal, to better narrow the scope of the recommendation.\n\nUsing a combination of these strategies could significantly improve the system’s ability to retrieve more targeted options."
}}

---

## Your Task:

### Inputs:
- Scratchpad: {Scratchpad}
- Original Response: {Original_response}
- Ground Truth Item Description: {Ground_truth}
- sample_k: {Sample_num}

### Output Format
{{
  "generative_reward": "<Your analysis and {Sample_num} improvement strategies written as a single natural language explanation>"
}}
"""

ASK_POTENTIAL_FUNCTION_PROMPT = """
You are a generative potential function specifically designed for the recommendation agent scenario. Your role is to produce natural language feedback analyzing the effectiveness of the original clarification question and suggesting how it could be improved.

You will receive the following inputs:

- The user's original input (`user_input`)
- The system’s original clarification question (`original_clarification`)
- A description of the intended target item (`ground_truth_item_description`)

Your task is to:
- Evaluate whether the original clarification question effectively encourages the user to provide more distinctive, retrievable details.
- Produce a concise, well-reasoned natural language analysis describing:
  1. Whether the original clarification was effective.
  2. How it could be improved to better align with the strategy.

⚠️ **Strict Constraints:**
1. You **must not** use, copy, infer, or incorporate any content from the ground truth item description in your suggestions. This includes:
   - Specific characters, settings, themes, narrative styles, or plot elements.
   - Any reverse-engineered implication of what the user "probably meant."
2. Your improvement suggestions must be **based solely on the user's original input.**
3. The output **must not** contain any hints or references to the ground truth description.

---

### Input Example

user_input:
> "I'm looking for a restaurant."

original_clarification:
> "Do you want something casual or fancy?"

ground_truth_item_description:
> Osteria Langhe — An intimate Northern Italian restaurant with low lighting and a romantic ambiance

---

### Example Output
{{
  "generative_reward": "The original question focuses only on a single binary dimension (casual or fancy) and asks a subjective preference, which does little to narrow down the search space. Since the user did not specify a location, it would be better to start by asking about location. In multi-turn contexts, consider prompting about other dimensions such as cuisine, price range, atmosphere, special requirements, or operating hours."
}}

---
## Your Task:

### Inputs:
- Scratchpad: {Scratchpad}
- Original Response: {Original_response}
- Ground Truth Item Description: {Ground_truth}

### Output Format
{{
  "generative_reward": "<your analysis of the original clarification question and suggestions for improvement>"
}}

"""


SEARCH_POTENTIAL_FUNCTION_PROMPT_v1 = """
You are a **generative potential function**, specifically designed for the retrieval submodule within a recommendation system.

Your task is to **evaluate the original search query for retrieval effectiveness and generate multiple rewrites to improve recall coverage or precision.**

---

You will receive the following inputs:

- The user's original input (`user_input`)
- The system's current generated search query (`original_query`)
- The intended target item description (`ground_truth_item_description`)
- The required number of candidates (`sample_k`)

---

Your responsibilities:

1. **Analyze whether the original query accurately captures the user’s intent and whether it is well-suited for the retriever.**
2. **Output a natural language analysis (`generative_reward`)** that highlights potential issues or strengths in the query and suggests possible improvements.
3. **Based on this analysis, generate `sample_k` alternative search queries** that may improve retrieval performance (in coverage or relevance).

---

### Types of Query Rewrites (can be combined):

- **Coverage-Oriented**: Retain all elements of the original intent; verbose but thorough.
- **Salience-Oriented**: Focus on only the most critical, distinctive parts of the intent; concise and precise.
- **Semantic Variants**: Use synonyms or related phrases to rephrase the query.
- **Structural Reordering**: Rearrange keywords or their groupings to change emphasis.
- **Contextual Expansion**: Add likely missing qualifiers or modifiers based on the user’s original intent.

---

⚠️ **Strict Constraints**:

1. All rewrite suggestions **must be keyword-style expressions** suitable for a retriever (e.g., keyword phrases, concise boolean expressions—not full natural language questions or sentences).
2. **You must not** use or reference any content from the target item description (including titles, plot elements, locations, character names, etc.).
3. All rewrites **must be derived solely from the user’s original input**, with no external assumptions or inference.

---

## Input Fields:

- Scratchpad: {Scratchpad}
- Original Search Query: {Original_response}
- Target Item Description: {Ground_truth}
- Number of Candidates (`sample_k`): {Sample_num}

---

## Output Format (must be valid JSON):

{{
  "generative_reward": "<Your evaluation of the current query and directions for improvement>",
}}
"""

SEARCH_POTENTIAL_FUNCTION_PROMPT = """
You are a **generative potential function** specifically designed for the recommendation agent scenario.

Your role is to **generate natural language feedback analyzing the effectiveness of the original search query and suggesting improvements.**

---

You will receive the following inputs:

- The user's original input (`user_input`)
- The system’s original search query (`original_query`)
- A description of the intended target item (`ground_truth_item_description`)

---

Your task is to:

- **Evaluate** whether the original search query effectively captures the user's intent and facilitates retrieval of relevant results.
- **Produce a concise, well-reasoned natural language analysis** describing:
  1. Whether the original search query is effective.
  2. How it could be improved to better locate the intended target.

Additionally, please include recommendations for **two improvement strategies**:

1. **Coverage-oriented**
   - Try to preserve all potentially relevant features mentioned by the user.
   - A more complete, verbose version of the user’s intent.
   - Good for exploratory search (more recall).

2. **Salience-oriented**
   - Focus only on the most discriminative, unique features in the user’s expression.
   - Aim for specificity and precision—shorter is better.
   - Good for targeted search (more precision).

---

⚠️ **Strict Constraints:**
1. You **must not** use, copy, infer, or incorporate any content from the ground truth item description in your suggestions. This includes:
   - Specific characters, settings, themes, narrative styles, or plot elements.
   - Any reverse-engineered implication of what the user "probably meant."
2. Your improvement suggestions **must be based solely on the user's original input.**
3. The output **must not** contain any hints or references to the ground truth description.

---
## Your Task:

### Inputs:
- Scratchpad: {Scratchpad}
- Original Response: {Original_response}
- Ground Truth Item Description: {Ground_truth}

### Output Format
{{
  "generative_reward": "<your analysis of the original search query and suggestions for improvement>",
}}
"""

RECOMMENDATION_POTENTIAL_FUNCTION_PROMPT_v1 = """
You are a **generative potential function** specifically designed for conversational recommendation agents.
Your task is to **analyze the system’s recommendation response, evaluate its effectiveness, and generate a natural language critique** that highlights both strengths and weaknesses.

---

You will be given the following inputs:

- The user's original input (`user_input`)
- The system's current recommendation output (`original_recommendation`)
- A list of items retrieved by the system (`retrieved_items`) — each is a string representing the exact retrieved item title
- The expected target item (`ground_truth_item`) — the one the system ideally should include
- The number of rewrite candidates desired (`sample_k`)

---

### Your task includes:

#### ✅ Recommendation Effectiveness Evaluation

1. **Content Coverage**:
   - Assess whether the recommendation **adequately covers the user's intent** and offers enough relevant options from the retrieved results.
   - If key retrieved items are omitted (especially the ground truth), treat this as a critical failure.

2. **Verbatim Matching**:
   - Check whether all recommended items exactly match their titles from `retrieved_items`.
   - **No edits are allowed** — including punctuation changes, space insertions, or paraphrasing.
   - Every recommended item must appear **verbatim**, or it will be considered incorrect.

3. **Flexibility**:
   - Was the system response overly rigid or one-dimensional?
   - Good recommendations should accommodate multi-aspect user inputs, balance exploration and specificity, and remain adaptable to different preference styles.

4. **Coherence**:
   - Does the recommendation read smoothly and make logical sense?
   - Critique awkward structure, unnatural transitions, or disjointed item listing.

5. **User Guidance**:
   - Evaluate whether the response helps the user take next steps:
     - Did it offer actionable follow-up options?
     - Did it invite preferences or clarification?
     - Did it support exploration?

---

### Structural Requirements:

- If the `ground_truth_item` **is in `retrieved_items` but not in the recommendation**:
  👉 This is a **coverage failure**. The response must include it verbatim in rewrites.

- If the `ground_truth_item` **is not in `retrieved_items`**, or is already included:
  👉 Focus on improving **language quality**, **flow**, and **engagement**, without changing the set of items.

---

### Strict Constraints:

1. You **must not use, infer, or reference** any content from `ground_truth_item_description` (e.g., plot, setting, characters, product features).
2. All recommended items must be drawn only from `retrieved_items`, using **exact title strings**.
3. You must not fabricate or paraphrase item titles.
4. You must not introduce speculative content based on the ground truth.
5. You may use connecting phrases to improve naturalness, but do not alter the item strings themselves.

---
## Your Task:

### Inputs:
- Scratchpad: {Scratchpad}
- Original Response: {Original_response}
- Ground Truth Item Description: {Ground_truth}
- Number of Candidates (`sample_k`): {Sample_num}


### Output Format

You must generate a single JSON object with the following field:

{{
  "generative_reward": "<Your analysis of the original recommendation’s issues and suggestions for how to rewrite it from multiple distinct directions (up to sample_num variants).>"
}}
"""

RECOMMENDATION_POTENTIAL_FUNCTION_PROMPT = """
You are a **generative potential function** specifically designed for the recommendation agent scenario.
Your role is to **produce natural language feedback analyzing the effectiveness of the original recommendation response and suggesting improvements.**

---

You will receive the following inputs:

- The user's original input (`user_input`)
- The system’s original recommendation response (`original_recommendation`)
- A description of the intended target item (`ground_truth_item_description`)

---

**Your task is to:**

- **Evaluate** whether the original recommendation effectively presented **all potentially relevant items** to the user.

- **Evaluate** whether the recommendation **faithfully preserved the exact text of the retrieved results**:
  The item names must be reproduced **verbatim from the original retrieval**, without any edits, optimizations, or omissions—even a single punctuation mark or version number must remain unchanged.
  We rely on **exact string matching** to verify correctness.
  > For example, if the retrieved item is:
  > "Title: UFOPETIE Card Case for Nintendo Switch Game Card,Compatible with Nintendo Switch Case Animal Crossing Theme,Game Case for Animal Crossing Cards,for Switch Accessories-A Leaf-Thumb Grip 2PCS...; Description"
  >
  > Then the recommendation **must exactly reproduce this text**, including all commas, spaces, and punctuation.
  >
  > If the output is:
  > "UFOPETIE Card Case for Nintendo Switch Game Card, Compatible with Nintendo Switch Case Animal Crossing Theme, Game Case for Animal Crossing Cards, for Switch Accessories - A Leaf-Thumb Grip 2PCS"
  >
  > Although it appears similar, it will still fail matching.

- **Produce a concise, well-reasoned natural language analysis** explaining:
  1. Whether the original recommendation was effective.
  2. How it could be improved to better cover the user's intent and ensure the target item is included verbatim.

---

⚠️ **Strict Constraints:**
1. You **must not** use, copy, infer, or incorporate any content from the ground truth item description in your suggestions. This includes:
   - Specific characters, settings, themes, narrative styles, or product details.
   - Any reverse-engineered implication of what the user "probably meant."
2. Your improvement suggestions **must be based solely on the user's original input and the recommendation response.**
3. The output **must not** contain any hints or references to the ground truth description.

---

### Example Output

{{
    "generative_reward": "The original recommendation only listed a few retrieved items and did not cover all relevant results. It also failed to preserve the exact format of the titles. To improve, the response should enumerate all retrieved items and include their text verbatim, maintaining all punctuation and spacing to ensure exact matching. Additionally, providing short descriptions for each item can help users compare options more confidently."
}}

---
## Your Task:

### Inputs:
- Scratchpad: {Scratchpad}
- Original Response: {Original_response}
- Ground Truth Item Description: {Ground_truth}

### Output Format
{{
  "generative_reward": "<your analysis of the original recommendation and suggestions for improvement>"
}}
"""

ASK_POLICY_IMPROVEMENT_PROMPT_abpotential = """
You are a strategy improvement operator, designed as a policy function specifically for the recommendation agent scenario.

Your task is to **generate {Sample_num} refined clarification questions** based on:
- the user's original input,
- the original clarification question,

Each refinement should:
✅ Avoid using or implying any details from the ground truth description.
✅ Be derived solely from the user's input and the improvement suggestions.
✅ Encourage the user to provide more distinctive, retrievable information.
✅ Use natural, open-ended, non-binary language.

---

### Input Example

user_input:
> "I'm looking for a restaurant."

original_clarification:
> "Do you want something casual or fancy?"

---

### Example Output
format: Ask[<your rewritten clarification question>]"

{{
  "refinement_output": [
    "Ask[Which city are you in? Do you have a specific price range in mind for the restaurant? Is there anything in particular you're looking for in a restaurant?]",
    ....
  ]
}}


## Your Task:

### Inputs:
- Scratchpad: {Scratchpad}
- Original Response: {Original_response}

### Output Format
{{
  "refinement_output": [
    "Ask[...]",
    ....
  ]
}}
"""

ASK_POLICY_IMPROVEMENT_PROMPT = """
You are a strategy improvement operator, designed as a policy function specifically for the recommendation agent scenario.

Your task is to **generate {Sample_num} refined clarification questions** based on:
- the user's original input,
- the original clarification question,
- and the generative reward containing suggestions for improvement.

Each refinement should:
✅ Avoid using or implying any details from the ground truth description.
✅ Be derived solely from the user's input and the improvement suggestions.
✅ Encourage the user to provide more distinctive, retrievable information.
✅ Use natural, open-ended, non-binary language.

---

### Input Example

user_input:
> "I'm looking for a restaurant."

original_clarification:
> "Do you want something casual or fancy?"

generative_reward:
> The original question focuses only on a single binary dimension (casual or fancy) and asks a subjective preference, which does little to narrow down the search space. Since the user did not specify a location, it would be better to start by asking about location. In multi-turn contexts, consider prompting about other dimensions such as cuisine, price range, atmosphere, special requirements, or operating hours.

---

### Example Output
format: Ask[<your rewritten clarification question>]"

{{
  "refinement_output": [
    "Ask[Which city are you in? Do you have a specific price range in mind for the restaurant? Is there anything in particular you're looking for in a restaurant?]",
    ....
  ]
}}


## Your Task:

### Inputs:
- Scratchpad: {Scratchpad}
- Original Response: {Original_response}
- Generative Reward: {Generative_reward}

### Output Format
{{
  "refinement_output": [
    "Ask[...]",
    ....
  ]
}}

"""


SEARCH_POLICY_IMPROVEMENT_PROMPT_abpotential = """
You are a strategy improvement operator, designed as a policy function specifically for the recommendation agent scenario.

Your task is to **generate {Sample_num} refined search queries** based on:
- the user's original input,
- the original search query,

Each refinement should:
✅ Avoid using or implying any details from the ground truth description.
✅ Be derived solely from the user's input and the improvement suggestions.
✅ Use natural, concise language optimized for semantic search.

---

## Your Task:

### Inputs:
- Scratchpad: {Scratchpad}
- Original Search Query: {Original_response}

### Output Format
{{
  "refinement_output": [
    "Search[...]",
    ...
  ]
}}
"""

SEARCH_POLICY_IMPROVEMENT_PROMPT = """
You are a strategy improvement operator, designed as a policy function specifically for the recommendation agent scenario.

Your task is to **generate {Sample_num} refined search queries** based on:
- the user's original input,
- the original search query,
- and the generative reward containing suggestions for improvement.

Each refinement should:
✅ Avoid using or implying any details from the ground truth description.
✅ Be derived solely from the user's input and the improvement suggestions.
✅ Use natural, concise language optimized for semantic search.

---

## Your Task:

### Inputs:
- Scratchpad: {Scratchpad}
- Original Search Query: {Original_response}
- Generative Reward: {Generative_reward}

### Output Format
{{
  "refinement_output": [
    "Search[...]",
    ...
  ]
}}
"""


RECOMMENDATION_POLICY_IMPROVEMENT_PROMPT = """
You are a strategy improvement operator, designed as a policy function specifically for the recommendation agent scenario.

Your task is to **generate {Sample_num} revised recommendation response** based on:
- the user's original input,
- the original recommendation response,
- and the generative reward containing suggestions for improvement.

Each refinement should:
✅ Enumerate **all retrieved items** without omission.
✅ Reproduce each item’s text **verbatim**, exactly as it appeared in retrieval results, including all punctuation, spaces, and formatting.
✅ Provide clear, concise reasoning for the recommendations.
✅ Encourage the user to engage or respond.

⚠️ Strict Constraints:
- You **must not** use or imply any details from the ground truth description.
- You **must** base your output solely on the user's input, the original recommendation, and the generative reward.
- You **must not** fabricate any additional items or modify the retrieved titles.

---

## Your Task:

### Inputs:
- Scratchpad: {Scratchpad}
- Original Recommendation: {Original_response}
- Generative Reward: {Generative_reward}

---

### Output Format
{{
  "refinement_output": [
    "Recommend[...]",
    ...
  ]
}}
"""




RECOMMENDATION_POLICY_IMPROVEMENT_PROMPT_abpotential = """
You are a strategy improvement operator, designed as a policy function specifically for the recommendation agent scenario.

Your task is to **generate {Sample_num} revised recommendation response** based on:
- the user's original input,
- the original recommendation response,

Each refinement should:
✅ Enumerate **all retrieved items** without omission.
✅ Reproduce each item’s text **verbatim**, exactly as it appeared in retrieval results, including all punctuation, spaces, and formatting.
✅ Provide clear, concise reasoning for the recommendations.
✅ Encourage the user to engage or respond.

⚠️ Strict Constraints:
- You **must not** use or imply any details from the ground truth description.
- You **must** base your output solely on the user's input, the original recommendation, and the generative reward.
- You **must not** fabricate any additional items or modify the retrieved titles.

---

## Your Task:

### Inputs:
- Scratchpad: {Scratchpad}
- Original Recommendation: {Original_response}

---

### Output Format
{{
  "refinement_output": [
    "Recommend[...]",
    ...
  ]
}}
"""

ASK_POTENTIAL_EVAL_PROMPT = """
You are a potential value estimator for an interactive recommendation agent.

We will provide you with:
- The agent's internal state (`s`)
- One original action (clarification question)
- {Sample_num} improved actions (rewritten clarifications)
- A description of the final target

Your task is to **evaluate from the user perspective** whether any improved action is more effective at eliciting useful, discriminative, and retrievable information compared to the original action.

Instructions:
1. Carefully compare the original action and each improved action.
2. If at least one improved action is significantly better, output:
   - `"is_better": true`
   - `"refinement_output"` containing the best improved action
3. If none of the improved actions are better, output:
   - `"is_better": false`
   - Do not provide `"refinement_output"`

⚠️ Strict Constraints:
- You **can reference** the target description to inform your evaluation
- Your output **must** be valid JSON.
- You should select the single best improved action that is most effective overall.

---

### Input Example

user_input:
> "I'm looking for a restaurant."

original_clarification:
> "Ask[Do you want something casual or fancy?]"

refinement_output:
> Ask[Which city are you in? Do you have a specific price range in mind for the restaurant? Is there anything in particular you're looking for in a restaurant?]

### Output Example
{{
  "is_better": true,
  "original_output": " {Original_response}",
  "refinement_output": "Ask[Which city are you in? Do you have a specific price range in mind for the restaurant? Is there anything in particular you're looking for in a restaurant?]"
}}

---

## Your Task:

### Inputs:
- Scratchpad: {Scratchpad}
- Original Response: {Original_response}
- Ground Truth: {Ground_truth}
- Refinement_output: {Refinement_output}

### Output Format
{{
  "is_better": true,
  "original_output": " {Original_response}",
  "refinement_output": "<the best improved action>"
}}
"""

RECOMMENDATION_POTENTIAL_EVAL_PROMPT = """
You are a potential value estimator for an interactive recommendation agent.

We will provide you with:
- The agent's internal state (`s`)
- One original action (recommendation response)
- {Sample_num} improved actions (rewritten recommendation responses)
- A description of the final target

Your task is to **evaluate from the user perspective** whether any improved recommendation is more effective at:
- Covering all retrieved items comprehensively
- Faithfully reproducing item titles exactly as retrieved
- Providing clear reasoning and encouraging user engagement

Instructions:
1. Carefully compare the original recommendation and each improved recommendation.
2. If at least one improved recommendation is significantly better, output:
   - `"is_better": true`
   - `"refinement_output"` containing the single best improved recommendation
3. If none of the improved recommendations are better, output:
   - `"is_better": false`
   - Do not provide `"refinement_output"`

⚠️ Strict Constraints:
- You **can reference** the target description to inform your evaluation.
- You **must** ensure that the recommended titles are reproduced verbatim.
- Your output **must** be valid JSON.


---

## Your Task:

### Inputs:
- Scratchpad: {Scratchpad}
- Original Response: {Original_response}
- Ground Truth: {Ground_truth}
- Refinement_output: {Refinement_output}

### Output Format
{{
  "is_better": true,
  "original_output": " {Original_response}",
  "refinement_output": "<the best improved action>"
}}
"""



ask_potential_function_template = PromptTemplate(
    input_variables=["Scratchpad", "Original_response", "Ground_truth", "Sample_num"],
    template=ASK_POTENTIAL_FUNCTION_PROMPT_v1
)

ask_policy_improvement_template = PromptTemplate(
    input_variables=["Scratchpad", "Original_response", "Generative_reward", "Sample_num"],
    template=ASK_POLICY_IMPROVEMENT_PROMPT
)

ask_policy_improvement_abpotential_template= PromptTemplate(
    input_variables=["Scratchpad", "Original_response", "Sample_num"],
    template=ASK_POLICY_IMPROVEMENT_PROMPT_abpotential
)


ask_potential_eval_template = PromptTemplate(
    input_variables=["Scratchpad", "Original_response", "Ground_truth", "Refinement_output", "Sample_num"],
    template=ASK_POTENTIAL_EVAL_PROMPT
)


search_potential_function_template = PromptTemplate(
    input_variables=["Scratchpad", "Original_response", "Ground_truth", "Sample_num"],
    template=SEARCH_POTENTIAL_FUNCTION_PROMPT_v1
)

search_policy_improvement_abpotential_template = PromptTemplate(
    input_variables=["Scratchpad", "Original_response", "Sample_num"],
    template=SEARCH_POLICY_IMPROVEMENT_PROMPT_abpotential
)

search_policy_improvement_template = PromptTemplate(
    input_variables=["Scratchpad", "Original_response", "Generative_reward", "Sample_num"],
    template=SEARCH_POLICY_IMPROVEMENT_PROMPT
)

recommendation_potential_function_template= PromptTemplate(
    input_variables=["Scratchpad", "Original_response", "Ground_truth", "Sample_num"],
    template=RECOMMENDATION_POTENTIAL_FUNCTION_PROMPT_v1
)

recommendation_policy_improvement_abpotential_template = PromptTemplate(
    input_variables=["Scratchpad", "Original_response", "Sample_num"],
    template=RECOMMENDATION_POLICY_IMPROVEMENT_PROMPT_abpotential
)

recommendation_policy_improvement_template = PromptTemplate(
    input_variables=["Scratchpad", "Original_response", "Generative_reward", "Sample_num"],
    template=RECOMMENDATION_POLICY_IMPROVEMENT_PROMPT
)

recommendation_potential_eval_template = PromptTemplate(
    input_variables=["Scratchpad", "Original_response", "Ground_truth", "Refinement_output", "Sample_num"],
    template=RECOMMENDATION_POTENTIAL_EVAL_PROMPT
)