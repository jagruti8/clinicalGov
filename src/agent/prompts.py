from langchain_core.prompts import ChatPromptTemplate

QUERY_CONSTRUCTOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Extract structured search parameters from the user's clinical trial question. "
        "Return ONLY valid JSON with these fields (null if not mentioned):\n"
        '- condition: disease or medical condition\n'
        '- intervention: drug, treatment, or procedure\n'
        '- phase: one of PHASE1, PHASE2, PHASE3, PHASE4\n'
        '- status: one of RECRUITING, COMPLETED, NOT_YET_RECRUITING\n\n'
        'Example: {{"condition": "type 2 diabetes", "intervention": "metformin", "phase": null, "status": "RECRUITING"}}'
    )),
    ("human", "{question}"),
])

REFORMULATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "The initial clinical trial search returned no relevant results. "
        "Generate improved parameters using synonyms, broader terms, or alternative drug names. "
        "Return ONLY valid JSON with: condition, intervention, phase, status (null if not applicable). "
        "Do NOT repeat the previous terms exactly."
    )),
    ("human", (
        "Original question: {question}\n"
        "Previous params (failed): {query_params}\n"
        "Attempt: {reformulation_count}\n\n"
        "New search parameters:"
    )),
])

SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a clinical research assistant. Answer using ONLY the provided trial data.\n"
        "Rules:\n"
        "1. Cite trial IDs inline as [NCT12345678] when referencing specific findings\n"
        "2. Mention phase, status, and outcomes when relevant\n"
        "3. Acknowledge conflicting findings across trials\n"
        "4. State clearly if the context is insufficient to answer\n"
        "5. Do NOT infer or assume relationships between drugs beyond what is explicitly stated — "
        "if a trial compares drug A vs drug B, do not describe it as combining them\n"
        "6. Use the exact wording from the trial description when describing what a trial studies\n"
        "7. Do not generalise across trials — describe each trial individually and accurately\n\n"
        "Think step by step: first identify which trials are relevant to the question, "
        "then reason through what each relevant trial says, "
        "then synthesize a cited answer."
    )),
    ("human", "Question: {question}\n\nClinical Trial Data:\n{context}\n\nAnswer:"),
])

JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert evaluator of clinical trial research summaries. "
        "Return ONLY valid JSON with:\n"
        '- faithfulness: float 0-1 (are all claims supported by the context?)\n'
        '- citation_accuracy: float 0-1 (are NCT IDs correctly attributed?)\n'
        '- faithfulness_reasoning: one-sentence explanation\n'
        '- citation_reasoning: one-sentence explanation'
    )),
    ("human", "Context:\n{context}\n\nAnswer to evaluate:\n{answer}\n\nJSON evaluation:"),
])
