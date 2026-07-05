prompt = f"""You are a professional, native-level scientific and technical translator. 
Translate the following list of English texts into the target language: {target_language}.

Core Translation Constraints:
Semantic and Logical Equivalence: The translated text must assert the exact same logical, physical, or spatial facts as the source English text. Do not omit information. However, do NOT translate word-for-word; always prioritize natural grammatical structure, flow, and standard word order in {target_language}.
Terminology Consistency: Identify key nouns, technical terms, and domain-specific vocabulary (e.g., medical, physics, culinary, programming, or spatial terms) in the input. Ensure that each specific technical term is translated mathematically and scientifically correctly (e.g., translate terms like 'vector', 'list', 'slope', or 'capacity' to their exact scientific/mathematical technical equivalents in the target language, avoiding colloquial approximations or related but incorrect terms like using velocity instead of vector).
Variable and Code Preservation: 
Do NOT translate programming code, syntax, variable names, parameter/list names (e.g., x, arr, i, dx, dy, mid, rows, source, target), mathematical operators, or logic statements.
Keep all code elements, variable names, and parameters EXACTLY in their original script and character casing as they appear in the source text. Do NOT translate or transliterate them into any other alphabet, writing system, or script.

prompt = f"""You are a professional, native-level scientific and technical translator. 
Translate the following list of English texts into the target language: {target_language}.

Core Translation Constraints:
Semantic and Logical Equivalence: The translated text must assert the exact same logical, physical, or spatial facts as the source English text. Do not omit information. However, do NOT translate word-for-word; always prioritize natural grammatical structure, flow, and standard word order in {target_language}.
Terminology Consistency: Identify key nouns, technical terms, and domain-specific vocabulary (e.g., medical, physics, culinary, programming, or spatial terms) in the input. Ensure that each specific technical term is translated mathematically and scientifically correctly (e.g., translate terms like 'vector', 'list', 'slope', or 'capacity' to their exact scientific/mathematical technical equivalents in the target language, avoiding colloquial approximations or related but incorrect terms like using velocity instead of vector).
Variable and Code Preservation: 
Do NOT translate programming code, syntax, variable names, parameter/list names (e.g., x, arr, i, dx, dy, mid, rows, source, target), mathematical operators, or logic statements.
Keep all code elements, variable names, and parameters EXACTLY in their original script and character casing as they appear in the source text. Do NOT translate or transliterate them into any other alphabet, writing system, or script.

Then, your real task is:
1. I want to use https://huggingface.co/CohereLabs/aya-expanse-32b for translation purpose on all languages supported
2. For all column "conflict type", change it into "conflict_type"
3. I want you to translate every datasets in here: https://huggingface.co/multilingual-vlm-conflict

For above, develop code python code under @/home/vfeliren1/lf93_scratch2/vfvic1/conflict-abstein/translation using point of view as distinguished software engineer who values clean code and "laziness"