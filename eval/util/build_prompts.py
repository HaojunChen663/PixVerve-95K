import json

# Build the prompt for Qwen3.5-35B-A3B global-scale evaluation (4 dimensions)
def build_global_eval_prompt():
    """
    Build the prompt for Qwen3.5-35B-A3B global-scale image fidelity evaluation.
    
    Returns:
         str: Structured JSON prompt string for MLLM.
    """

    prompt = f"""
# ROLE AND TASK FORMULATION:
You are a professional image quality assessment (IQA) and scoring expert specializing in T2I evaluation. Your task is to carefully evaluate the **global-scale fidelity** of the provided synthetic image across 4 dimensions. Please focus on the overall composition and structure, geometric and physical logic, and macro-scale consistency.

# CRITICAL SCORING RULES (Must Strictly Follow):
1. **Objectivity & Fairness:** Maintain an objective stance throughout the evaluation process and base your judgement on visual evidence with the same standard instead of subjective preference.
2. **Focus Solely on Fidelity:** Consider the image category and expected characteristics while avoiding any bias towards the content of the image. Score based on the visual quality and fidelity aspects solely.
3. **Independence:** Evaluate each dimension independently without any halo effects.
4. **Rigor:** Apply strict criteria and any noticeable artifact should be reflected in the scoring. Maintain a high standard for what constitutes a "5" (Excellent).

# EVALUATION RUBRICS:
## 1. Structural Coherence (SC-global)
Check whether the geometric structure of the entities is correct, whether there are any missing or redundant limbs, and whether the overall spatial relations are consistent with physical common sense.
- **5 (Excellent):** Flawless physical logic. Objects are perfectly formed; no missing/extra parts.
- **4 (Good):** Minor structural oddities that don't distract from the main subject.
- **3 (Fair):** Noticeable structural errors (e.g., slightly deformed limbs or merged background objects).
- **2 (Poor):** Obvious physical failures; objects are partially collapsed or mutated.
- **1 (Very Poor):** Severe structural collapse; chaotic composition; unrecognizable forms.

## 2. Perspective Integrity (PI)
Check whether the perspective relationship between objects at different distances conforms to the principles of perspective, and whether there is any distorted perspective.
- **5 (Excellent):** Flawless perspective and geometric projection. Vanishing points and horizon lines align accurately.
- **4 (Good):** Slight perspective tilt, but geometric projection still feels natural.
- **3 (Fair):** Distorted depth; objects at different distances feel "stacked" or misaligned.
- **2 (Poor):** Severe geometric warping; architectural lines curve unnaturally or conflict.
- **1 (Very Poor):** Multiple conflicting vanishing points or warped architectural lines; total perspective failure.

## 3. Lighting Consistency (LC)
Check whether the overall lighting has consistency with that of a natural image, and whether there are obviously artificial brightness gradients.
- **5 (Excellent):** Unified light source. Shadows, highlights, and reflections follow ray-tracing logic.
- **4 (Good):** Consistent lighting, but subtle mismatch in shadow softness or intensity.
- **3 (Fair):** Ambiguous light source. Some objects appear "self-lit" without casting shadows.
- **2 (Poor):** Contradictory lighting directions. Shadows cast in different ways for nearby objects.
- **1 (Very Poor):** Complete lighting failure; flat "sticker-like" objects with zero interaction with the environment.

## 4. Color Harmony (CH)
Check whether the overall color transitions are smooth and natural, and whether there are issues such as blurred edges of color blocks and color banding.
- **5 (Excellent):** Natural color gamut. Smooth gradients; no banding or abnormal color noise.
- **4 (Good):** High-quality color, though slight over-saturation or other issues in small areas.
- **3 (Fair):** Visible color banding in gradients. Slight chromatic aberrations.
- **2 (Poor):** Patchy color blocks; unnatural "neon" artifacts or gray/dull patches in vibrant areas.
- **1 (Very Poor):** Severe color corruption; massive chromatic noise or broken color channels.

# OUTPUT RULES (Must Strictly Follow):
1. You MUST follow a strict 5-point scale and provide a score as an **INTEGER from 1 to 5 only** for each dimension.
2. Provide the final output strictly **in a JSON object inside the <json> tag**.
3. The <json> block MUST contain ONLY the valid JSON object. No markdown code blocks or extra text.
4. Keys: "SC-global", "PI", "LC", "CH" represent the scores for the 4 dimensions respectively, and "reasoning" is a concise explanation justifying the scores. Ensure the JSON property names are enclosed in double quotes and there are no trailing commas in the JSON object.

# OUTPUT FORMAT:
<json>
{{
  "SC-global": int,
  "PI": int,
  "LC": int,
  "CH": int,
  "reasoning": "A concise explanation (about 2-4 sentences) justifying the four-dimensional scores."
}}
</json>

**Output example for reference (do not copy this exact content, just an example of the structure):**
<json>
{{
  "SC-global": 4,
  "PI": 5,
  "LC": 5,
  "CH": 5,
  "reasoning": "The image demonstrates excellent global fidelity with natural perspective and consistent, warm lighting that suggests a late afternoon setting. The color palette is harmonious and transitions smoothly without banding. However, minor structural imperfections are visible in the fine details of the hands and fingers, which appear slightly indistinct or merged, preventing a perfect score in structural coherence."
}}
</json>

Now, please evaluate the **global-scale fidelity** of the provided image based on the above criteria and output the scores and reasoning in the specified **JSON** format.
    """
    return prompt.strip()

# Build the prompt for Qwen3.5-35B-A3B local-scale evaluation (5 dimensions)
def build_local_eval_prompt(relative_coords):
    """
    Build the prompt for Qwen3.5-35B-A3B local-scale image fidelity evaluation.
    
    Inputs:
         relative_coords (list): The normalized patch coordinates [x_min, y_min, x_max, y_max] relative to the global image, where (0,0) is the top-left corner and (1,1) is the bottom-right corner.
    Returns:
         str: Structured JSON prompt string for MLLM.
    """
    
    prompt = f"""
# ROLE AND TASK FORMULATION:
You are a professional image quality assessment (IQA) and scoring expert specializing in local fine-grained details evaluation. Your task is to carefully evaluate the **local-scale fidelity** of the provided **local patch** of an ultra-high-resolution synthetic image across 5 dimensions. To ensure high-quality evaluation, please adhere to the following guidelines:

# IMPORTANT CONTEXT:
- **Image 1 (First):** This is the **LOCAL PATCH (target for scoring)**. Ignore incomplete objects or composition issues. Focus only on the visual quality of the visible area.
- **Image 2 (Second):** This is the **FULL GLOBAL IMAGE (for contextual reference)**. **DO NOT** use this image for direct visual analysis or evaluation. Use this image ONLY for **understanding the original image's theme and global context**.
- **Patch Location:** The local patch (Image 1) corresponds to the area defined by the relative coordinates {relative_coords} in the global image (Image 2). The relative coordinates are normalized [x_min, y_min, x_max, y_max], where (0,0) is the top-left corner and (1,1) is the bottom-right corner of the global image.

# CRITICAL SCORING RULES (Must Strictly Follow):
1. **Objectivity & Fairness:** Maintain an objective stance throughout the evaluation process and base your judgement on visual evidence with the same standard instead of subjective preference.
2. **Focus Solely on Fidelity:** Consider the image category and expected characteristics while avoiding any bias towards the content of the image. Score based on the visual quality and fidelity aspects solely.
3. **Local-to-Global Evaluation:** Evaluate the details in Image 1, and use Image 2 to distinguish between "intended bokeh/blur" and "accidental artifacts".
4. **Coordinates Reference:** Use the rectangular bounding box only to understand the local patch's location in the overall image context, but DO NOT directly compare the local patch to the global image for pixel-level details.
5. **Independence:** Evaluate each dimension independently without any halo effects.
6. **Rigor:** Apply strict criteria and any noticeable artifact should be reflected in the scoring. Maintain a high standard for what constitutes a "5" (Excellent).

# EVALUATION RUBRICS:
Please evaluate the microscopic details and fidelity of the **Local Patch (Image 1)** across the 5 dimensions below, while using the Global Image (Image 2) and the relative coordinates {relative_coords} as reference.
## 1. Noise and Grain Existence (NGE)
Check whether there is random high-frequency color noise and obvious color graininess in the local patch.
- **5 (Excellent):** Crystal clean and realistic cinematic grain. Zero digital noise or compression blocks. Grain looks like natural film if present.
- **4 (Good):** Slight luminance noise, barely visible at 100% zoom.
- **3 (Fair):** Noticeable noise or grain that distracts from the details, especially in shadow areas.
- **2 (Poor):** Heavy salt-and-pepper noise or distracting grain.
- **1 (Very Poor):** Image details are buried under severe noise or compression corruption.

## 2. Generative Artifacts (GA)
Check whether typical generative artifacts (e.g., checkerboard artifacts and edge halos) are present in the local patch.
- **5 (Excellent):** No AI-specific artifacts. Details look photo-realistic and like they were captured by a high-end CMOS sensor.
- **4 (Good):** Minor generative patterns that require close inspection to find.
- **3 (Fair):** Noticeable AI "melting" or "waxy" textures where details should be sharp.
- **2 (Poor):** Hallucinated textures or "ghosting" artifacts typical of diffusion models.
- **1 (Very Poor):** Massive generative collapse; "AI-soup" textures.

## 3. Texture Fidelity (TF)
Check whether the local patch presents plastic-like oversmoothing, and for natural objects such as wood and fabric, whether the texture has randomness rather than mechanical repetition.
- **5 (Excellent):** Tactile realism. Skin pores, fabric weaves, or surface grit are ultra-sharp and authentic.
- **4 (Good):** High detail, but slightly "soft" or over-regularized texture.
- **3 (Fair):** Texture is visible but "flat"; lacks the micro-depth of real-world surfaces.
- **2 (Poor):** Over-smoothed "plastic" look; details are smeared out.
- **1 (Very Poor):** Completely blurred or "mushy" surfaces with zero recognizable texture.

## 4. Micro-geometry Coherence (MGC)
Check at a local scale whether the lines show unacceptable jitter or jagged edges.
- **5 (Excellent):** Perfect edge continuity. Fine lines (e.g., hair, wires) are smooth at the pixel level.
- **4 (Good):** Sharp edges, though very minor aliasing (stair-stepping) visible on diagonals.
- **3 (Fair):** Jagged edges or slight shimmering; fine lines appear broken in some places.
- **2 (Poor):** Severe aliasing; pixelated edges; micro-structures look "broken".
- **1 (Very Poor):** Total geometric chaos at the micro-level; edges are unrecognizable.

## 5. Sharpness Consistency (SC-local)
Check whether there are unnatural blurry patches (i.e., within the same focal plane some areas is clear enough while others are abnormally blurry).
- **5 (Excellent):** Natural optical sharpness variation consistent with depth of field. No inconsistent blur patches within the same focal plane.
- **4 (Good):** Slightly soft, but no inconsistent blur patches within the same focal plane.
- **3 (Fair):** Noticeable inconsistency in sharpness; some areas look artificially sharpened while others are blurry without a natural depth-of-field reason.
- **2 (Poor):** Obvious sharpness inconsistency; "cut-and-paste" feel where the patch looks like it was taken from a different image with different focus.
- **1 (Very Poor):** Severe sharpness failure; the patch looks like a low-quality thumbnail pasted into the global image.

# OUTPUT RULES (Must Strictly Follow):
1. You MUST follow a strict 5-point scale and provide a score as an **INTEGER from 1 to 5 only** for each dimension.
2. Provide the final output strictly **in a JSON object inside the <json> tag**.
3. The <json> block MUST contain ONLY the valid JSON object. No markdown code blocks or extra text.
4. Keys: "NGE", "GA", "TF", "MGC", "SC-local" represent the scores for the 5 dimensions respectively, and "reasoning" is a concise explanation justifying the scores. Ensure the JSON property names are enclosed in double quotes and there are no trailing commas in the JSON object.

# OUTPUT FORMAT:
<json>
{{
  "NGE": int,
  "GA": int,
  "TF": int,
  "MGC": int,
  "SC-local": int,
  "reasoning": "A concise explanation (about 2-4 sentences) justifying the five-dimensional scores."
}}
</json>

**Output example for reference (do not copy this exact content, just an example of the structure):**
<json>
{{
  "NGE": 3,
  "GA": 2,
  "TF": 3,
  "MGC": 4,
  "SC-local": 4,
  "reasoning": "The local patch exhibits noticeable grain in the shadow areas, particularly on the floor, preventing a higher noise score. A significant generative artifact is present where the hand holding the tablet appears to be a mechanical claw with red fingers, which is inconsistent with the person in the full global image. The paint texture on the floor appears slightly waxy and overly regular, lacking the random imperfections of real liquid paint, though the overall sharpness and edge coherence are consistent with the scene's depth of field."
}}
</json>

Now, please evaluate the **local-scale fidelity** of the provided local crop based on the above criteria and output the scores and reasoning in the specified **JSON** format.
    """
    return prompt.strip()


# Build the prompt for Qwen3.5-35B-A3B ICS evaluation (3 dimensions)
def build_ICS_eval_prompt(long_caption):
    """
    Build the prompt for Qwen3.5-35B-A3B instance-centric compliance evaluation.

    Inputs:
         long_caption (str): The long caption used to generate the image.
    Returns:
         str: Structured JSON prompt string for MLLM.
    """
    prompt = f"""
# ROLE AND TASK FORMULATION:
You are a professional image quality assessment and scoring expert specializing in Text-to-Image (T2I) semantic alignment evaluation. Your task is to carefully perform a fine-grained, instance-centric assessment of the provided synthesized image based on its initial detailed long caption.

**Input Long Caption:** "{long_caption}"

# METRIC DIMENSIONS AND BOUNDARIES:
Please evaluate the image across the following three distinct, hierarchical dimensions, strictly adhering to the defined criteria and boundaries:
1. **IEV (Instance Existence Verification):** Inspect whether all instances explicitly mentioned in the long caption are present. Focus strictly and solely on presence or absence rather than quality.
2. **AAA (Appearance Attribute Alignment):** For each instance that exists, assess whether its visual attributes (color, texture, material, size, shape) align with the description in the long caption. This requires detailed cross-referencing between the caption and the visual content.
3. **SRA (Spatial Relation Accuracy):** Evaluate whether the relative positioning (e.g., left/right, top/bottom, foreground/background) and the logical perspective between multiple instances are accurately depicted in the image.

# CRITICAL SCORING RULES (Must Strictly Follow):
1. **Hierarchical Dependence:** **IEV** is the gatekeeper. If any critical instance is missing (IEV below 4), the corresponding AAA and SRA for the image must be penalized accordingly, as attributes and relations cannot exist without the entity.
2. **Detail Awareness:** Since this is a high-resolution image evaluation task, you must meticulously scan **the entire canvas**, including corners and background, to identify all mentioned instances and their micro-details.
3. **Strict Adherence to Explicit Constraints:** Judge the image ONLY based on what is explicitly stated in the long caption. Do not impose imaginary constraints or personal aesthetic preferences. For any visual aspects NOT mentioned (e.g., specific lighting, background nuances, or artistic style), the generation model is allowed creative autonomy. Do not penalize the model for "making choices" where the prompt is silent.
4. **Hallucination Penalty:** If the synthesized image contains prominent instances that are NOT mentioned in the long caption and significantly distract from the caption's content (severe hallucination), deduct 1-2 points from **IEV**.
5. **No Middle Ground Bias:** Avoid giving 7/10 by default. Be decisive based on the visual evidence.
6. **Objectivity & Fairness:** Maintain an objective stance throughout the evaluation process and base your judgement on visual evidence with the same standard instead of subjective preference.

# SCORING RUBRICS (10-Point Scale):
## 1. IEV (Instance Existence Verification)
- **[9-10]:** All instances (primary and secondary) from the long caption are present and clearly identifiable in the image. No significant omissions.
- **[7-8]:** Primary instances are present; only minor, non-essential background elements are missing or extremely unclear.
- **[5-6]:** At least one primary instance is missing, or multiple instances are severely obscured; but the image still captures the general theme and content of the caption.
- **[3-4]:** Multiple primary instances are missing or unrecognizable, significantly detracting from the caption's description.
- **[1-2]:** Total mismatch; the image fails to depict the core content of the caption.

## 2. AAA (Appearance Attribute Alignment)
- **[9-10]:** Perfect alignment. All visible attributes (color, texture, material, size, shape) of the instances matches the caption exactly.
- **[7-8]:** Most attributes are correct; slight deviations in secondary details (e.g., shade of color or minor texture mismatch) that do not significantly affect the overall perception.
- **[5-6]:** Noticeable attribute misalignment in primary instances (e.g., wrong color or material), but the image still somewhat reflects the caption's content.
- **[3-4]:** Severe attribute misalignment; primary instances look totally different from the text description.
- **[1-2]:** Complete attribute failure; objects lack any descriptive fidelity or are rendered as generic blobs.

## 3. SRA (Spatial Relation Accuracy)
- **[9-10]:** All relative positions between instances and depth cues perfectly match the spatial prepositions in the caption.
- **[7-8]:** Relative positions are correct, but there are minor errors in scale/perspective or overlapping logic that do not cause major confusion.
- **[5-6]:** Noticeable spatial relation errors (e.g., left/right flipped, foreground/background confusion).
- **[3-4]:** Obvious spatial relation failures; instances are positioned in a way that contradicts the caption's logic.
- **[1-2]:** Chaotic layout; objects are floating or positioned randomly without regard for the caption's logic.

# OUTPUT RULES (Must Strictly Follow):
1. You MUST follow a strict 10-point scale and provide a score as an **INTEGER from 1 to 10 only** for each dimension.
2. Provide the final output strictly **in a JSON object inside the <json> tag**.
3. The <json> block MUST contain ONLY the valid JSON object. No markdown code blocks or extra text.
4. Keys: "reasoning", "IEV", "AAA", and "SRA". "reasoning" is a concise explanation justifying the scores; "IEV", "AAA", and "SRA" represent the scores for the 3 dimensions respectively. Ensure the JSON property names are enclosed in double quotes and there are no trailing commas in the JSON object.

**Output example for reference (do not copy this exact content, just an example of the structure):**
<json>
{{
  "reasoning": "The cat and the bench is present but the straw hat is missing; the absence of the hat results in a deduction in IEV. The cat's color and texture are reasonably well-aligned with the caption, though the fur appears more orange than described, leading to a moderate AAA score. The spatial relations are mostly accurate, with the cat sitting on bench and positioned on the left as specified, but the background elements are somewhat jumbled, resulting in a high but not perfect SRA score.",
  "IEV": 6,
  "AAA": 8,
  "SRA": 9
}}
</json>

Now, please evaluate the **instance-centric compliance** of the provided image based on the above criteria and output the scores and reasoning in the specified **JSON** format.
    """
    return prompt.strip()