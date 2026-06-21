# ======== 新的Refiner Prompt =======

# ==> Typing 部分
TYPE_ERROR_SYSTEM_PROMPT = """
You are a professional Named Entity Recognition (NER) annotator.

Your task is to perform **Entity Type Refinement**.  
Given a text and its **Initial Entity List**, you must correct any incorrect entity types according to the provided definitions.

- You must **only correct the entity types**. 
- You must **not change the text spans**.
- You must **preserve the original order** of the entities.
- Some entities may already be correctly typed and do not need changes.
- Always output the result as a **list of [text span, type] pairs**.

If no changes are needed, simply output the entity list as it is.

You must strictly follow the Entity Type Definitions provided.
"""

TYPE_ERROR_PROMPT = """
# Task Instruction
Your task is to refine entity types in an entity list extracted from the text.

- You are provided with the text and its Initial Entity List.
- You must check whether the entity types are correct according to the definitions.
- Some entities are already correct and should remain unchanged.
- Some entities have incorrect types and need to be corrected.
- Only modify the entity types when necessary.
- Always output a complete entity list as a list of [text span, type] pairs, preserving the original order.

# Entity Type Definitions
{type_definitions}

# Examples
{examples}

-----

Text: {input_text}
Initial Entity List: {input_entity_list}
Refined Entity List:
"""

# =====> Spurious Entities 部分 ======

SPURIOUS_SYSTEM_PROMPT = """
You are a very cautious NER–annotation refiner.
Your ONLY job: identify entities in the provided list "Entities" of the given "Text" that are **certainly** spurious.
*Delete an entity only when you are very sure it is wrong.*

Deleting a correct entity is a serious error.  
When in doubt, KEEP the entity.

## Strict Criteria:
- A **spurious entity** is one that is **clearly incorrect and cannot reasonably be interpreted** as a valid entity given the definitions and context.
- You should **only mark an entity as spurious if you are highly confident it is wrong**.
- If you are **unsure or the entity seems at least partially plausible**, **do NOT delete it**.
- When in doubt, **always choose to retain the entity** rather than remove it.

## Constraints:
- You must **only select from the provided Entities list**. 
- Do **not** invent, rephrase, or modify any entities.
- Do **not** add new entities or explanations.

## Output Format:
- If there are spurious entities: [[text1, type1], [text2, type2], ...]
- If there are no spurious entities: output the word `None` (case-sensitive, no quotes).
"""


SPURIOUS_ERROR_PROMPT = """
# Task Instruction
You are a professional NER annotation refiner specializing in detecting **spurious entities**.

Your task is to review the provided "Entities" list based on the given "Text", and identify any **spurious entities** — entities that are **clearly and unquestionably incorrect** based on the context and the provided entity type definitions.

## Entity Type Definitions
{type_definitions}

# Guidelines
- A **spurious entity** is one that does **not clearly represent a meaningful, valid entity**.
- Only flag an entity as spurious if it is **obviously incorrect** and has **no plausible justification** under the given definitions.
- If you are **unsure** or the entity could be **reasonably interpreted as valid**, **do NOT delete it**.
- Be **very conservative** in your judgment: **only remove with high confidence**.
- Only select from the original list. Do not invent or modify anything.

# Output Format:
- If any entities are clearly spurious: [[text1, type1], [text2, type2], ...]
- If none are spurious: return the word `None` (case-sensitive, no quotes).
- Do not include any comments or explanations.

# Examples
{examples}

-----

Text: {input_text}
Entities: {input_entities}
Spurious Entities:
"""




# ==> Missing Error 部分 ⬇️⬇️⬇️
MISSING_SYSTEM_PROMPT = """
You are an expert NER refiner.  
Given a *Text* and its current *Entities* list, return **only** the entities that are truly missing.

Adding an incorrect or questionable entity is a **serious error**.  
You must be extremely cautious. It is better to **miss a valid entity** than to introduce a wrong one.

Guidelines:
1. A "missing entity" is a valid span in the text that should be annotated but is absent from *Entities*.  
2. Propose an entity **only if you are certain**; **when in doubt, skip it**.  
3. Do **not** output any span already present, nor overlapping / duplicate spans.  
4. **Output must be a bare list — e.g. [[text1, type1], [text2, type2]] — or the literal word None.**

Output Format:
- If some are missing: [[text1, type1], [text2, type2], …]  
- If none are missing: None
"""

MISSING_ERROR_PROMPT = """
# Task Instruction
You are an expert NER refiner for **missing entities**. 
Carefully examine the given *Text* and its current *Entities*. Your job is to identify valid entities that are **missing** — i.e., present in the text but completely absent from *Entities*.

Adding an incorrect or questionable entity is a **serious error**.  
You must be extremely cautious. It is better to **miss a valid entity** than to introduce a wrong one.

## Entity Type Definitions
{type_definitions}

# Guidelines
- A missing entity is a span that should have been annotated but is **entirely missing**.
- **Only include entities that you are highly confident are valid and missing.**
- If a span is borderline, ambiguous, or just "possible", **leave it out**.
- Do **not** output:
   - Any entity already in the list.
   - Any span that overlaps, contains, or is contained by an existing span.
   - Any entity that might be a near-duplicate or boundary variant of an existing one.
- Be extremely conservative. **Adding spurious or questionable entities is worse than missing a few.**

# Output Format:
- If some are missing: [[text1, type1], [text2, type2], …] 
- If only one is missing: [[text, type]]
- If none: output the word `None` (case-sensitive, no quotes)

# Examples  
{examples}

-----

Text: {input_text}
Entities: {input_entities}
Missing Entities:
"""



# ==> Span Error 部分
SPAN_SYSTEM_PROMPT = """
You are a careful NER annotation refiner.

Your task is to review the "Initial Annotated Text" and correct **only the most obvious span boundary errors** by slightly adjusting entity spans using XML-style tags:
`<entity type="TYPE">text</entity>`

Guidelines:
- Do NOT modify any non-entity text.
- Do NOT change entity types.
- Do NOT add or delete entities.
- Do NOT make changes unless the span is clearly wrong (e.g., includes extra text, misses important words, or merges two entities).
- If you are unsure, keep the original span unchanged.

Output Format:
- Output the full "Refined Annotated Text" in valid XML-style annotation.
- Do not add any explanations.
"""

SPAN_ERROR_PROMPT = """
# Task Instruction
You are a careful NER annotation refiner.

Your task is to review the "Initial Annotated Text" and correct **only the most obvious span boundary errors** by slightly adjusting entity spans using XML-style tags:
`<entity type="TYPE">text</entity>`

# Guidelines
- Do NOT modify any non-entity text.
- Do NOT change entity types.
- Do NOT add or delete entities.
- Do NOT make changes unless the span is clearly wrong (e.g., includes extra text, misses important words, or merges two entities).
- If you are unsure, keep the original span unchanged.

# Output Format
- Output the full "Refined Annotated Text" in valid XML-style annotation.
- Do not add any explanations.

# Examples
Below are examples showing span refinement. Some entities require small boundary adjustments, while many are already correct and should be preserved exactly.

{examples}

--------

Initial Annotated Text: {input_annotated_text}
Refined Annotated Text:
"""





# TYPE_ERROR_PROMPT = """
# You are reviewing Named Entity Recognition (NER) annotations for {dataset} domain text, focusing exclusively on entity type accuracy. Relevant entity types: {labels}

# # Entity Type Definitions
# {type_definitions}

# # Task Instruction
# Identify **Type Errors** - cases where an entity's assigned type is clearly wrong in context. 

# Key Rules:
# 1. **Scope**: Only check type assignments, NOT spans/boundaries or missing entities
# 2. **Threshold**: Flag ONLY when:
#    - The text clearly belongs to a different type in the definitions
#    - The context makes the correct type unambiguous
# 3. **Default**: Keep original annotation if type is even slightly plausible


# Output Format:
# - If no errors, just output "No type errors found."
# - If errors are found, list all potential type errors in the format: 
#     - <entity type="annotated_type">text</entity> may have wrong type. The type is more likely [suggested_type].
#     - ...

# # Examples
# {examples}


# Text: {input_text}
# Annotated Text: {input_annotated_text}
# Output:"""

# TYPE_ERROR_PROMPT = """
# # Task Instruction
# You are an expert quality assurance specialist reviewing Named Entity Recognition (NER) annotations for text from the {dataset} domain. Your task is to identify potential **Type Errors** within the provided 'Input Annotated Sample'.

# A **Type Error** occurs when the `type` assigned to an entity's `text` span seems semantically incorrect or inconsistent, considering the 'Original Text' context and the specific rules and entity types defined in the 'Annotation Guidelines and Entity Type Definitions'. The original annotations use an XML-style format like `<entity type="TYPE">text</entity>`.

# Focus *exclusively* on identifying **Type Errors**. Do **not** report issues related to annotation spans (completeness/boundaries), missing entities, or entities that seem generally spurious unless their type assignment is the core issue making them invalid according to the guidelines.

# Use the provided 'Annotation Guidelines' and 'Examples' to understand the correct application of entity types like {labels}.

# # **Definition of Type Error**
# An annotation should be flagged as a **Type Error** if the `text` span, interpreted within the `Original Text` context, does not align semantically with the definition or examples provided for its assigned `type` in the entity type definition. Pay attention to specific distinctions required by the guidelines.

# # Annotation Guidelines and Entity Type Definitions
# ```
# {annotation_guidelines}
# ```

# # Examples
# Here are relevant examples of **correct** annotations based on the 'Annotation Guidelines and Entity Type Definitions':
# {examples}

# # Input Annotated Sample
# {input_annotated_text}

# # Your Findings
# Please review the 'Input Annotated Sample' provided above. Identify all instances where an `<entity>` tag has a `type` attribute that seems inconsistent with the tagged text content, based *only* on the 'Annotation Guidelines', context, and 'Examples'. Focus *only* on **Potential Type Errors**.

# **Based on your review, provide your response in one of the following two structured natural language formats ONLY:**

# **Format 1: If NO Potential Type Errors are found:**
# No entity type error found.

# **Format 2: If one or more Potential Type Errors ARE found:**
# Start your response exactly with the line: `Yes. The potential errors:`
# Then, on subsequent lines, list each potential type error you identified as a bullet point starting with `- `(a hyphen followed by a space). Example Response (Format 2):
# ```
# Yes. The potential errors:
# - "<entity type="annotated_type">text</entity>" is more like a "[suggested_type]" entity rather than "[annotated_type]". Since [brief justification based on guidelines/context]
# - ...
# ```   

# Choose only ONE of the formats above for your entire response. Do not add introductory or concluding remarks. Your entire output must strictly follow one of these two structures.
# """

# SPAN_ERROR_PROMPT = """
# # Task Instruction
# You are an expert quality assurance specialist reviewing Named Entity Recognition (NER) annotations for text from the {dataset} domain. The interested entity types are {labels}. Your task is to identify potential **Span Errors** within the provided 'Input Annotated Sample'.

# The annotations are embedded directly within the text using XML-style tags like `<entity type="TYPE">text</entity>`. A **Potential Span Error** occurs when the text content *inside* an `<entity>` tag seems incorrectly bounded – either too narrow (incomplete) or too broad (includes extraneous text) – based on the 'Original Text' context and the 'Annotation Guidelines'.

# Focus *exclusively* on identifying **Potential Span Errors**. Do **not** report errors related to the entity `type` itself, missing entities, or spurious annotations unless the core issue is the boundary of the tagged text span.

# Use the provided 'Annotation Guidelines' and 'Examples' to understand the correct rules for defining annotation boundaries.

# # **Definition of Potential Span Error**
# An annotation (represented by an `<entity>` tag) should be flagged as a **Potential Span Error** if the text boundaries of the content *inside* the tag do not accurately capture the intended entity according to the rules in the `Annotation Guidelines`. Check for:
# * **Incompleteness:** The span is too short and misses parts of the entity name/term (e.g., tagging only "Hinton" instead of "Geoffrey Hinton" if guidelines require full names).
# * **Over-extension:** The span is too long and includes surrounding words that are not part of the entity itself (e.g., tagging "University of Pennsylvania located in..." instead of just "University of Pennsylvania").
# * **Boundary Awkwardness:** The span doesn't align well with natural linguistic units based on the guidelines.

# # Annotation Guidelines and Entity Type Definitions
# ```
# {annotation_guidelines}
# ```

# # Examples
# # Here are relevant examples demonstrating correct annotations based on the 'Annotation Guidelines':
# {examples}

# # Input Annotated Sample
# {input_annotated_text}

# # Your Findings
# Please review the 'Input Annotated Sample' provided above. Identify all instances where the text span inside an `<entity>` tag seems incorrectly bounded (too short, too long, awkward boundaries) based *only* on the 'Annotation Guidelines', context, and 'Examples'. Focus *only* on **Potential Span Errors**.

# **Based on your review, provide your response in one of the following two structured natural language formats ONLY:**

# **Format 1: If NO Potential Span Errors are found:**
# No entity span error found.

# **Format 2: If one or more Potential Span Errors ARE found:**
# Start your response exactly with the line: `Yes. The potential span errors:`
# Then, on subsequent lines, list each potential type error you identified as a bullet point starting with `- `(a hyphen followed by a space). Example Response (Format 2):
# ```
# Yes. The potential errors:
# - "<entity type="annotated_type">text</entity>" seems to have a span boundary error. It should likely be "[suggested_span]". Since [brief justification based on guidelines/context]
# - ...
# ```   

# Choose only ONE of the formats above for your entire response. Do not add introductory or concluding remarks. Your entire output must strictly follow one of these two structures.
# """

# MISSING_ENTITY_PROMPT = """
# # Task Instruction
# You are an expert quality assurance specialist reviewing Named Entity Recognition (NER) annotations for text from the {dataset} domain. Your task is to identify potential **Missing Entities** within the provided 'Input Annotated Sample'.

# The existing annotations are embedded directly within the text using XML-style tags (`<entity type="TYPE">text</entity>`). A **Potential Missing Entity** is a span of text that is currently **not** enclosed in any `<entity>` tag, but *should* have been annotated according to the 'Annotation Guidelines and Entity Type Definitions' and the context.

# Focus *exclusively* on identifying **Potential Missing Entities**. Scan the entire 'Input Annotated Sample', paying attention to un-tagged portions. Do **not** report errors in existing tags (like type or span errors) or suggest spurious annotations. Only report clear omissions based on the guidelines.

# Use the provided 'Annotation Guidelines' and 'Examples' to understand which types of entities should typically be captured and annotated.

# # **Definition of Potential Missing Entity**
# A text span within the 'Input Annotated Sample' should be flagged as a **Potential Missing Entity** if:
# 1.  It is **not** currently enclosed within any `<entity>` tags.
# 2.  It clearly represents an entity matching one of the definitions in the `Annotation Guidelines and Entity Type Definitions` (e.g., it's an entity falls under the defined types: {labels}).
# 3.  It appears relevant and meets any salience criteria mentioned in the guidelines (e.g., it's a specific named entity required by the guidelines).

# *Focus on clear and unambiguous omissions based on the provided guidelines.*

# # Annotation Guidelines and Entity Type Definitions
# ```
# {annotation_guidelines}
# ```

# # Examples
# Here are relevant examples demonstrating text snippets with correctly identified entities, illustrating the types of entities expected to be annotated:
# {examples}

# # Input Annotated Sample
# {input_annotated_text}

# # Your Findings
# Please review the entire 'Input Annotated Sample' provided above, including text outside of existing `<entity>` tags. Identify all clear instances where a span of text matching a defined entity type in the 'Annotation Guidelines and Entity Type Definitions' appears to be missing an annotation tag. Focus *only* on **Potential Missing Entities**.

# **Based on your review, provide your response in one of the following two structured natural language formats ONLY:**

# **Format 1: If NO Potential Missing Entities are found:**
# No missing entity error found.

# **Format 2: If one or more Potential Missing Entities ARE found:**
# Start your response exactly with the line: Yes. The potential missing entities:
# Then, on subsequent lines, list each potential missing entity you identified as a bullet point starting with - (a hyphen followed by a space). Example Response (Format 2): 
# ```
# Yes. The potential missing entities:
# - Text "[missing_text]" seems to be a missing entity that should be tagged as "[suggested_type]". Since [brief justification based on guidelines/context]
# - ...
# ```   

# Choose only ONE of the formats above for your entire response. Do not add introductory or concluding remarks. Your entire output must strictly follow one of these two structures.
# """


# SPURIOUS_ENTITY_PROMPT = """
# # Task Instruction
# You are an expert quality assurance specialist reviewing Named Entity Recognition (NER) annotations for text from the {dataset} domain. Your task is to identify potential **Spurious Entities** within the provided 'Input Annotated Sample'.

# The annotations are embedded directly within the text using XML-style tags (`<entity type="TYPE">text</entity>`). A **Potential Spurious Entity** is an *existing* annotation tag that should **not** have been made because the tagged text does not represent a valid, relevant, or required entity according to the 'Annotation Guidelines and Entity Type Definitions'.

# Focus *exclusively* on identifying **Potential Spurious Entities**. Look for annotations where the tagged text:
# * Does not meaningfully fit the definition of the assigned type, nor any other valid type defined in the guidelines.
# * Represents a concept explicitly excluded by the guidelines.
# * Does not meet relevance or salience criteria defined in the guidelines (e.g., tagging overly generic terms).

# Do **not** report errors where the type or span is merely incorrect but the text *could* represent *some* valid entity if corrected. The focus is on annotations that should likely be **removed entirely**.

# Use the provided 'Annotation Guidelines' and 'Examples' to understand what should be annotated as a valid entity.

# # **Definition of Potential Spurious Entity**
# An existing annotation (represented by an `<entity>` tag) should be flagged as a **Potential Spurious Entity** if the tagged text span, even considering its context, does not represent a concept that should be annotated according to the `Annotation Guidelines`. This might be because it's not a specific named entity (if required), it belongs to an explicitly excluded category, it's too generic, or it fundamentally doesn't fit any defined entity type meaningfully based on the provided schema: {labels}.

# # Annotation Guidelines and Entity Type Definitions
# ```
# {annotation_guidelines}
# ```

# # Examples
# Here are relevant examples demonstrating text snippets with correctly annotations:
# {examples}

# # Input Annotated Sample
# {input_annotated_text}

# # Your Findings
# Please review the *existing* `<entity>` tags within the 'Input Annotated Sample' provided above. Identify all instances where an annotation seems invalid or unnecessary according to the 'Annotation Guidelines' and should likely be removed entirely. Focus *only* on **Potential Spurious Entities**.

# **Based on your review, provide your response in one of the following two structured natural language formats ONLY:**

# **Format 1: If NO Potential Spurious Entities are found:**
# No spurious entity error found.

# **Format 2: If one or more Potential Spurious Entities ARE found:**
# Start your response exactly with the line: Yes. The potential spurious entities:
# Then, on subsequent lines, list each potential spurious entity you identified as a bullet point starting with - (a hyphen followed by a space). Example Response (Format 2):
# ```
# Yes. The potential spurious entities:
# - "<entity type="TYPE">text</entity>" seems spurious and should likely be removed. Since [brief justification based on guidelines/context]
# - ...
# ```

# Choose only ONE of the formats above for your entire response. Do not add introductory or concluding remarks. Your entire output must strictly follow one of these two structures.
# """



# ========== MODIFIER PROMPTS ==========

MODIFIER_PROMPT = """
# Task Instruction
You are an intelligent NER annotation editor for text from the {dataset} domain. Your task is to refine the annotations in the 'Annotated Sample' based on a 'Judge Report' containing potential errors. The annotations use XML-style tags (`<entity type="TYPE">text</entity>`).

Your goal is to produce a final, corrected version of the annotated text by critically evaluating the Judge's findings against the 'Annotation Guidelines' and applying only the necessary corrections directly to the XML string.

# Annotation Guidelines
--- START OF ANNOTATION GUIDELINE ---
{annotation_guidelines}
--- END OF ANNOTATION GUIDELINE ---

# Annotation Examples (Illustrating Correct Annotation)
The following are relevant examples of **correct** annotations based on the 'Annotation Guidelines'. Use these to understand the target annotation style and how guidelines should be applied correctly.

{examples}

# Annotated Sample and Judge Report
The following report summarizes findings from four separate checks: Type, Span, Missing, and Spurious errors of the "Annotated Sample". Review each section. Sections starting with "Yes." contain potential errors to evaluate against the guidelines. Sections starting with "No..." indicate no errors of that specific type were found by the judge. 

**Annotated Sample**:
{input_annotated_text}

**Judge Report**:
{judge_report}

# Your Task & Action Steps
Carefully perform the following steps:
1. Thoroughly understand the `Annotation Guidelines` and review the `Annotation Examples`.
2. Read the `Annotated Sample` and `Judge Report`.
3. **Critically Validate Each Finding:** For every potential error mentioned in the `Judge Report` (if any):
    a. Locate the relevant entity/text in the `Annotated Sample`.
    b. Evaluate the Judge's suggestion and justification strictly against the `Annotation Guidelines`. Do **not** blindly trust the Judge. Perform your own validation based on the rules.
    c. Decide if a correction (Type change, Span change, Insertion, Deletion) is truly warranted according to the guidelines.
4. **Plan & Apply Edits:** Determine the exact modifications needed for all *validated* errors. Apply these edits directly to the `Annotated Sample` string. Ensure the XML structure remains valid and surrounding text is preserved. Handle potential overlapping or interacting edits carefully.
5. **Generate Output:** Prepare the final XML-tagged text string. If the Judge reported no errors OR if your validation determined no corrections were needed according to the guidelines, this string will be the original `Annotated Sample`. If you planned valid corrections in step 4, this string will be the modified text incorporating *all* validated changes. Follow the output structure below *exactly*.


# Final Annotation Output:
This is the original text you need to modify based on your validation of the judge report:
**Annotated Sample**: {input_annotated_text}

Now, generate the final output below this line.
IMPORTANT: Ensure your final output is the complete text with well-formed XML tags throughout.
- If the Judge reported no errors OR if your validation determined no corrections were needed according to the guidelines, output the original **Annotated Sample** exactly as shown above.
- If you identified valid corrections based on the guidelines, output the complete, modified XML-tagged text incorporating *all* validated changes.
Your entire response after the 'Final Annotated Sample' label must ONLY be the final XML-tagged text, with no extra explanations.
**Final Annotated Sample**:
"""


# Renaming to indicate Chain-of-Thought version
MODIFIER_PROMPT_COT = """
# Task Instruction
You are an intelligent NER annotation editor for text from the {dataset} domain. Your task is to refine the annotations in the 'Annotated Sample' based on a 'Judge Report' containing potential errors. The annotations use XML-style tags (`<entity type="TYPE">text</entity>`).

Your goal is to first provide an analysis by critically evaluating the Judge's findings against the 'Annotation Guidelines', and then produce a final, corrected version of the annotated text by applying only the necessary corrections directly to the XML string.

# Annotation Guidelines
--- START OF ANNOTATION GUIDELINE ---
{annotation_guidelines}
--- END OF ANNOTATION GUIDELINE ---

# Annotation Examples (Illustrating Correct Annotation)
The following are relevant examples of **correct** annotations based on the 'Annotation Guidelines'. Use these to understand the target annotation style and how guidelines should be applied correctly, which will help you validate the Judge's findings.
{examples}

# Annotated Sample and Judge Report
The following report summarizes findings from four separate checks: Type, Span, Missing, and Spurious errors of the "Annotated Sample". Review each section. Sections starting with "Yes." contain potential errors to evaluate against the guidelines. Sections starting with "No..." indicate no errors of that specific type were found by the judge.

**Annotated Sample**:
{input_annotated_text}

**Judge Report**:
{judge_report}

# Your Task & Action Steps
Carefully perform the following steps:
1. Thoroughly understand the `Annotation Guidelines` and review the `Annotation Examples`.
2. Read the `Annotated Sample` and `Judge Report`.
3. **Critically Validate Each Finding:** For every potential error mentioned in the `Judge Report` (if any):
    a. Locate the relevant entity/text in the `Annotated Sample`.
    b. Evaluate the Judge's suggestion and justification strictly against the `Annotation Guidelines`. Do **not** blindly trust the Judge. Perform your own validation based on the rules.
    c. Decide if a correction (Type change, Span change, Insertion, Deletion) is truly warranted according to the guidelines. Keep track of your reasoning.
4. **Plan Edits:** Determine the exact modifications needed for all *validated* errors based on your decisions in Step 3.
5. **Generate Output:** Based on your validation (Step 3) and planned edits (Step 4), first generate your detailed **Analysis** explaining your decisions for each reported potential error. Then, generate the **Final Annotated Sample** reflecting all the valid corrections you applied (or the original text if no corrections were made). Follow the specific two-part output structure defined below *exactly*.

# Your Response (Analysis and Final Annotatation Output)
**First, provide your step-by-step analysis, starting exactly with `Analysis:`.** Detail your evaluation of each potential error listed in the `Judge Report` (mention which section it came from - Type, Span, etc., if helpful). For each point you evaluate:
    - State clearly whether you **agree** or **disagree** with the Judge's finding based on your interpretation of the `Annotation Guidelines` and context.
    - Briefly explain your reasoning, referencing specific guidelines if possible.
    - If you agree and will make a correction, state the specific correction you intend to apply.
    - If the Judge reported no errors, or if you reviewed all reported errors and deemed none valid, state that clearly in your analysis.

**Second, after your complete analysis, provide the final, complete annotated text starting exactly on a new line with `Final Annotated Sample:`.** This should be the potentially modified XML-tagged text reflecting all the corrections you decided to make based on your analysis. Apply the edits carefully, ensuring the XML structure remains valid and surrounding text is preserved. If no corrections were ultimately made based on your validation, this section should contain the original `Annotated Sample` unchanged.
**IMPORTANT:** Ensure the final output XML is well-formed. Do not include explanations after the `Final Annotated Sample:` label. Your entire response after this label must be only the final XML-tagged text. 


# Analysis and Final Annotation Output
This is the original text you need to modify based on your validation of the judge report:
**Input Annotated Sample**: {input_annotated_text}

Now, generate your response based on the detailed instructions provided in the '# Your Response (Analysis and Final Output)' section above:
- Remember to critically validate the Judge Report against the Guidelines and perform edits carefully.
- Ensure the 'Final Annotated Sample' uses complete and well-formed XML tags.
- Use the following two-part structure precisely, providing only the requested content after each label:

**Analysis:**
[Your detailed analysis evaluating Judge findings, justifying decisions based on guidelines, and stating intended corrections or noting no valid errors found]

**Final Annotated Sample**:
[Your final complete XML-tagged text, either original or modified according to your analysis]
"""




ANNOTATION_PROMPT = """
You are a professional annotation guideline writer for Named Entity Recognition (NER) tasks.

The target text domain is {dataset}.

The interested entity types are: {labels}

You are given a set of NER-annotated examples where entities are marked using XML-style tags like:
<entity type="...">...</entity>

Your task is to analyze these examples and generate comprehensive annotation instructions for a human annotator or a language model to replicate the annotation process.

For each entity type, please provide:
1. **Label Definitions**: Define each entity type based on how it is used in the examples.
2. **Annotation Patterns**: Describe typical linguistic patterns, cues, or contextual clues that indicate the presence of each entity type. Include example phrases or syntactic structures if useful to illustrate how and when to annotate.

Here are the annotated examples:
{examples}

Please generate clear, consistent, and practical instructions about the task instruction (label definition and annotation patterns) that can be followed to annotate new text in the same style.
Please follow the below format:

- "Type1"
Definition:
Patterns:
- "Type2"
Definition:
Patterns:
....
"""

ERROR_ANALYSIS_PROMPT = """
You are a professional error analysis expert for Named Entity Recognition (NER) tasks.
The target text domain is {dataset}.
The interested entity types are: {labels}
You are given a set of example pairs, one is the model prediction another is the reference gold standard. Both are NER-annotated examples where entities are marked using XML-style tags like:
<entity type="...">...</entity>
Your task is to analyze these prediction errors and polish the given annotation guideline for a human annotator or a language model to replicate the annotation process.

Here are the error examples:
{examples}
Please generate clear, consistent, and practical instructions about the error patterns for each entity type that can be followed to annotate new text in the same style.
Please just change the "Patterns" part of each entity type and "General Annotation Rules" in the annotation guideline.
Original annotation guideline:
--------------
{annotation_guideline}
--------------
"""


# ========== PROMPTS FOR LLM-as-Judge NER Annotation ======

REFINER_PROMPT = """
You are a Named Entity Recognition (NER) annotation specialist refining entity tags for the {dataset} domain. 

# Entity Type Guidelines
Available types: {labels}
Definitions:
{type_definitions}

# Refinement Protocol
Analyze each annotation for these potential errors:
1. **Type Error** - Entity type doesn't match definition/context
2. **Span Error** - Text boundaries are incorrect (over/under-spanning)
3. **Missing Entity** - Clear entity instance wasn't tagged
4. **Spurious Entity** - Non-entity text was incorrectly tagged

# Correction Rules
- Only make changes when errors are CLEAR and UNAMBIGUOUS
- If no errors are found, RETURN THE ORIGINAL ANNOTATION WITHOUT CHANGES
- For borderline cases, always prefer the original annotation
- Changes must be minimal and precisely targeted
- For corrections:
  1. Type errors: Update only the type attribute
  2. Span errors: Adjust only the text span
  3. Missing entities: Add new tags with minimal required span
  4. Spurious entities: Remove the tag completely

# Output Specifications
Return EXACTLY ONE OF:
1. The FULL TEXT with refined annotations in original XML format (if changes made)
2. The ORIGINAL ANNOTATION WITHOUT CHANGES (if no errors found)

# Examples
{examples}



Text: {input_text}
Current Annotation: {input_annotated_text}
Refined Annotation:
"""