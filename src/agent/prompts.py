from langchain_core.prompts import ChatPromptTemplate

TOOL_ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You route questions about clinical trials to tools. Pick the tool(s) that fetch the "
        "data needed to answer. Return ONLY valid JSON, no prose:\n"
        '{{"tool_calls": [{{"name": "<tool>", "args": {{...}}}}]}}\n\n'
        "Tools:\n{tools}\n\n"
        "Rules:\n"
        "- Use get_trial, get_eligibility, or compare_trials only when the question names NCT IDs.\n"
        "- Otherwise use search_trials. Omit args you do not need. Use at most 3 calls.\n"
        "- phase is one of PHASE1, PHASE2, PHASE3, PHASE4. "
        "status is one of RECRUITING, COMPLETED, NOT_YET_RECRUITING.\n\n"
        'Example: {{"tool_calls": [{{"name": "search_trials", "args": '
        '{{"condition": "type 2 diabetes", "phase": "PHASE3", "status": "RECRUITING"}}}}]}}'
    )),
    ("human", "{question}"),
])

PLAN_REPAIR = (
    "Your previous reply was not usable: {problem}. Reply again with ONLY a JSON object of the form "
    '{{"tool_calls": [{{"name": "<tool>", "args": {{...}}}}]}}. '
    "Valid tools: {tools}. Each call needs one of those names and valid args."
)

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
        "7. Do not generalise across trials — describe each trial individually and accurately\n"
        "8. Cite ONLY NCT IDs that appear in the provided trial data\n"
        "9. Present a trial as an answer only if its listed conditions and eligibility fit the question. "
        "If a trial studies a different population (for example prediabetes when asked about type 2 "
        "diabetes), say so plainly or leave it out\n\n"
        "Think step by step: first identify which trials are relevant to the question, "
        "then reason through what each relevant trial says, "
        "then synthesize a cited answer."
    )),
    ("human", "Question: {question}\n\nClinical Trial Data:\n{context}\n{feedback}\nAnswer:"),
])

REVISION_FEEDBACK = (
    "\nA reviewer found problems with your previous answer. Fix every one of them, "
    "and remove anything the trial data does not support:\n{issues}\n"
)

EVIDENCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "List every factual claim in the ANSWER that is attributed to a trial. "
        "Return ONLY valid JSON:\n"
        '{{"claims": [{{"claim": "<short claim>", "nct_id": "<NCT id>", '
        '"quote": "<exact words copied from that trial\'s data, at most 25 words>"}}]}}\n'
        "The quote must be copied verbatim from the trial data, not paraphrased."
    )),
    ("human", "Trial data:\n{context}\n\nANSWER:\n{answer}\n\nJSON:"),
])

CRITIQUE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You review answers about clinical trials against the trial data. Check for:\n"
        "- claims the data does not support\n"
        "- wrong phase, status, condition, or intervention relationships\n"
        "- citations to NCT IDs that are not in the data\n"
        "- failure to answer the question\n"
        "Return ONLY valid JSON: "
        '{{"verdict": "pass" or "revise", "issues": ["<specific problem>", ...]}}\n'
        "Use pass with an empty issues list if the answer is correct and supported."
    )),
    ("human", (
        "Question: {question}\n\nTrial data:\n{context}\n\nAnswer:\n{answer}\n\n"
        "Problems already found by automatic checks:\n{code_issues}\n\nJSON:"
    )),
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
    ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer to evaluate:\n{answer}\n\nJSON evaluation:"),
])
