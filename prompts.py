# ======= Prpmpts for LLM Error Aanalysis =======
analysis_prompt = """
# Task: 
Analyze Named Entity Recognition (NER) errors in LLM predictions compared to reference annotations and generate a detailed error report to improve future predictions. All entities are XML-style tagged with <entity type="TYPE">text</entity> format.

# Input:
You will receive:
- Prediction samples (LLM output with potential errors)
- Reference samples (correct gold standard annotations)

# Error Analysis Guidelines:

1. Comparison Criteria:

- Verify both entity spans (text boundaries) and entity types
- Identify:
    - Correct predictions (matching span and type)
    - Type mismatches (correct span, wrong type)
    - Span errors (incorrect text boundaries)
    - Missing entities (present in reference but not prediction)
    - False positives (present in prediction but not reference)

2. Error Categorization:
- Classify each error into one of the above categories
- Note patterns in errors (e.g., certain types frequently confused)
- Identify systematic issues (e.g., over-tagging, under-tagging)

3. Report Structure:
- Detailed examples of each error type
- Analysis of error patterns
- Specific recommendations for improvement to help LLM identify the potential errors
"""

enhance_annotation_guideline = """
## Enhanced NER Annotation Guidelines (Based on Error Analysis)

### Critical Focus Areas for Improved Accuracy:
1. **Technical Term Classification**:
   - Be particularly careful when classifying technical terms among <algorithm>, <task>, <product>, <field>, and <miscellaneous>
   - <algorithm> should only be used for specific, named computational procedures (e.g., "random forest", "k-means")
   - <task> should describe activities or objectives (e.g., "object detection", "sentiment analysis")
   - <field> should describe broad domains (e.g., "machine learning", "computer vision")
   - When in doubt between technical categories, prefer <miscellaneous>

2. **Miscellaneous Entities**:
   - Tag descriptive technical phrases as <miscellaneous> (e.g., "non-linear layouts", "pre-processing step")
   - Include conceptual terms under <miscellaneous> (e.g., "feature vectors", "acoustic models")
   - Historical/commemorative references should be <miscellaneous> (e.g., "60th anniversary of Turing's death")

3. **Entity Boundaries**:
   - Keep institutional names as single entities (e.g., "US Department of Commerce NIST")
   - Maintain complete technical terms as single entities (e.g., "Bilingual evaluation understudy metric")
   - Don't combine distinct entities (e.g., keep "Rajabazar Science College" and "University of Calcutta" separate)

4. **Over-tagging Prevention**:
   - Avoid tagging mathematical symbols (e.g., "β", "Σ")
   - Don't tag generic technical terms unless they're specifically named entities
   - Be conservative with standalone technical words (e.g., "capsule", "hyperplane")

5. **Specific Category Clarifications**:
   - Programming languages: Tag clearly as <programming language> (e.g., "Logo", "OCTAVE")
   - Locations: Distinguish carefully between:
     - <country> (nation states)
     - <location> (cities, physical locations)
     - <university> (academic institutions with place names)
   - Research entities: Clearly separate:
     - <organization> (research institutions, companies)
     - <university> (academic institutions)

### Examples for Reference:
Correct:
- "CRF-based tokenizer" → <product>
- "anomalous propagation" → <miscellaneous>
- "Cambridge, Massachusetts" → <university>
- "cluster analysis" → <task>
- "machine learning algorithms" → <field>

Incorrect:
- "OCR" as <field> → should be <task>
- "Michigan" as <country> → should be <location> or <university> depending on context
- "Optimization techniques" as <algorithm> → should be <field>
- "text-based systems" as <miscellaneous> → should not be tagged
"""







# ===== PROMPTS FOR CrossNER Benchmark =====
ONE_STAGE_TAG_FEW_SHOT_PROMPTS = """Task Description:
Identify and tag entities in the input text by wrapping them with XML-like tag <entity type="[TYPE]">...</entity> tags. Replace [TYPE] with the correct entity category. Make sure that you only wrap text that corresponds to an entity and leave the rest of the text unchanged. Tag all valid entities, even if they repeat.
Entity Types:
{entity_types}

Examples:
{examples}

Test Data:
Input: {input_text}
Output:
"""

# ========= PROMPTS FOR Zero Shot =========

ZERO_SHOT_TAG_PROMPT = """
Task Description:
Identify and tag entities in the input text by wrapping each entity with an XML-like tag. For every entity you detect, enclose the entity text in a tag using the following format:
    <entity type="[TYPE]">Entity Text</entity>
Replace [TYPE] with the appropriate category from the list below.

Entity Types:
{entity_types}

Output Format:
- Only wrap text segments that correspond to valid entities.
- Leave non-entity text unchanged.
- Tag all valid entities, even if they appear more than once.
- Ensure the output strictly follows the format, with matching opening and closing tags and no extra modifications.

Task:
Input: {input_text}
"""


ONE_STAGE_FEW_SHOT_PROMPTS = """Task Description: 
Identify and extract all entities from the input text along with their entity types. Return the results as a list of [entity, type] pairs. Ensure that all valid entities are captured, even if they repeat.
Entity Types:
{entity_types}

Examples:
{examples}

Task:
Input: {input_text}
"""



