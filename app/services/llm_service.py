import httpx
from typing import Optional, List
from langchain.llms.base import LLM
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from langchain.prompts import PromptTemplate
from app.core.config import settings
from app.utils.logger import setup_logger
from app.utils.tokenizer import count_tokens

logger = setup_logger(__name__)

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


class CustomLLM(LLM):
    """Custom LLM wrapper for the external API"""
    
    @property
    def _llm_type(self) -> str:
        return "custom"
    
    async def _acall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Async call to LLM API"""
        payload = {
            "model": "openai/gpt-oss-20b",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 2000
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(settings.LLM_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Sync call - not used but required by LangChain"""
        raise NotImplementedError("Use async call instead")


class LLMService:
    async def generate_insights_direct(
        self,
        chunks: List[str],
        case_type: str,
        data_source: str
    ) -> str:
        """Generate insights using direct LLM call (for small datasets)"""
        formatted_prompt = PROMPT_TEMPLATE.format(
            case_type=case_type,
            data_source=data_source
        )
        
        concatenated_data = "\n\n".join(chunks)
        user_message = f"{formatted_prompt}\n\nData to analyze:\n\n{concatenated_data}"
        
        payload = {
            "model": "openai/gpt-oss-20b",
            "messages": [{"role": "user", "content": user_message}],
            "temperature": 0.3,
            "max_tokens": 2000
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(settings.LLM_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    
    async def generate_insights_map_reduce(
        self,
        chunks: List[str],
        case_type: str,
        data_source: str
    ) -> str:
        """Generate insights using LangChain's map-reduce approach"""
        logger.info("Using map-reduce approach for large dataset")
        
        formatted_prompt = PROMPT_TEMPLATE.format(
            case_type=case_type,
            data_source=data_source
        )
        
        documents = [LangchainDocument(page_content=chunk) for chunk in chunks]
        llm = CustomLLM()
        
        map_template = f"""{formatted_prompt}

Analyze the following text segment and extract preliminary insights:

{{text}}

PRELIMINARY INSIGHTS:"""
        
        map_prompt = PromptTemplate(template=map_template, input_variables=["text"])
        
        reduce_template = f"""[Role]
You're an investigative insight generator consolidating multiple analysis reports.

[Task]
Merge the following preliminary insights into a single comprehensive report following this exact schema:
- Summary
- Criminal Activities
- Suspicious Keywords
- Connections to Leads
- Deeper Insights
- Classification

Case Type: {case_type}
Data Source: {data_source}

Preliminary insights to consolidate:

{{text}}

CONSOLIDATED INSIGHTS:"""
        
        reduce_prompt = PromptTemplate(template=reduce_template, input_variables=["text"])
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.MAX_TOKENS_PER_MAP_CHUNK * 3,
            chunk_overlap=200,
            length_function=count_tokens
        )
        
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(split_docs)} documents for map-reduce")
        
        chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=reduce_prompt,
            verbose=True
        )
        
        result = await chain.ainvoke({"input_documents": split_docs})
        return result["output_text"]