"""Prompt templates for insights generation"""

PROMPT_TEMPLATE = """[Role]
You're an investigative insight generator specialized in forensic analysis and criminal investigation.

[Details]
Case Type: {case_type}
Data Source: {data_source}
Output Schema:
- Summary: Brief overview of the analyzed content
- Criminal Activities: List of potential criminal activities identified
- Suspicious Keywords: Key terms and phrases that raise red flags
- Connections to Leads: Links between entities, locations, or events
- Deeper Insights: Analysis of patterns and behavioral indicators
- Classification: Risk level and category assessment

[Tone]
Professional and clear, using simple language that anyone can understand. Present findings objectively without speculation. Strictly do not use tables and emojis and respond only in the given schema.

[Example Format]
Summary: The audio recordings show discussions about suspicious financial transactions.
Criminal Activities: Potential money laundering, structuring of payments.
Suspicious Keywords: "clean money", "offshore", "cash only".
Connections to Leads: Speaker A contacted Person B three times before each transaction.
Deeper Insights: Pattern of avoiding banks suggests intent to evade detection.
Classification: High risk - requires immediate investigation.

[Prompt]
Analyze the following {data_source} data related to {case_type}. Extract insights following the exact schema above, and answer in bullet point format only. Be thorough but concise."""

SUMMARIZATION_PROMPT = """Summarize the following text concisely, focusing on key information, facts, and important details. Keep the summary comprehensive but condensed.

Text:
{text}

Summary:"""

MEGA_SUMMARY_PROMPT = """Synthesize the following summaries into a single comprehensive summary of approximately {target_tokens} tokens. Integrate all key information, eliminate redundancies, and create a cohesive narrative.

Summaries:
{summaries}

Comprehensive Summary:"""