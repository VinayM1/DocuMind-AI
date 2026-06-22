import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

client = OpenAI(timeout=30.0)

@dataclass
class RetrievalResult:
    case: Any
    score: float

@dataclass
class GeneratedInsight:
    content: Any
    confidence: float
    supporting_case_ids: List[str]
    insight_type: str

class CaseInsightGenerator:
    def __init__(self, model_name: str = "gpt-5"):
        self.model_name = model_name
        self.llm_initialized = bool(os.getenv("OPENAI_API_KEY"))
        self.token_counts = []
        

        self.prompt_templates = {
            'analysis': """*Task*: Analyze legal cases to answer: "{query}"
            
            *Cases* (sorted by relevance):
            {case_summaries}
            
            *Rules*:
            1. Use ONLY case content - no external knowledge
            2. Do NOT include any case IDs or references in your response
            3. Maximum 75 words
            4. Provide clear, direct analysis without technical references
            
            *Output*: Concise analysis without case IDs""",
            
            'recommendation': """*Role*: Insurance/legal expert assistant
            *Task*: Respond to query using ONLY the context below
            
            *Context*:
            {case_summaries}
            
            *Query*: {query}
            
            *Output Format* (JSON ONLY):
            {{
                "response_type": "decision|info|procedure|contact",
                "decision": "approved/rejected/N/A",
                "amount": "₹.../Covered/N/A",
                "answer": "1-sentence summary",
                "detailed_explanation": ["bullet 1", "bullet 2"],
                "referenced_clauses": ["Clause X.X"],
                "next_steps": ["action 1"],
                "baqis_score": 0-100
            }}
            
            *Rules*:
            1. Use ONLY context - if not found, use "N/A"
            2. baqis_score = confidence (0-100)
            3. MAX 5 bullet points
            4. Strict JSON format - NO additional text
            5. Do NOT include any case IDs or technical references in the response"""
        }

    def generate(self, query: str, retrieved_cases: List[RetrievalResult], insight_type: str = "analysis") -> GeneratedInsight:
        if not self.llm_initialized:
            return GeneratedInsight(
                content="LLM not initialized" if insight_type != "recommendation" else {"error": "LLM not initialized"},
                confidence=0.0,
                supporting_case_ids=[],
                insight_type=insight_type
            )

        case_summaries = self._prepare_case_summaries(retrieved_cases[:3])
        
        prompt = self.prompt_templates[insight_type].format(
            query=query,
            case_summaries=case_summaries
        )
        
        self.token_counts.append(len(prompt.split()))
        logger.info(f"Prompt tokens: {len(prompt.split())}")

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "You output valid JSON for recommendation type without any case IDs or technical references" if insight_type == "recommendation" 
                                   else "You provide concise legal analysis without case IDs or technical references"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower for more deterministic output
                max_tokens=500,
                response_format={"type": "json_object"} if insight_type == "recommendation" else None
            )
            
            llm_output = response.choices[0].message.content
            logger.info(f"LLM response: {llm_output[:100]}...")
            
            if insight_type == "recommendation":
                try:
                    llm_output = json.loads(llm_output)
                    llm_output = self._clean_case_ids_from_response(llm_output)
                except json.JSONDecodeError:
                    logger.error("JSON parse failed, attempting repair")
                    llm_output = self._repair_json(llm_output)
            else:
                llm_output = self._clean_case_ids_from_text(llm_output)
        
        except Exception as e:
            logger.error(f"OpenAI error: {str(e)}")
            llm_output = {"error": "Generation failed"} if insight_type == "recommendation" else "Generation failed"

        verification = self._verify_insight(
            str(llm_output) if insight_type != "recommendation" else json.dumps(llm_output),
            retrieved_cases
        )

        return GeneratedInsight(
            content=llm_output,
            confidence=verification['confidence'],
            supporting_case_ids=verification['supporting_ids'],
            insight_type=insight_type
        )

    def _prepare_case_summaries(self, cases: List[RetrievalResult]) -> str:
        summaries = []
        for i, res in enumerate(sorted(cases, key=lambda x: -x.score), 1):
            content = res.case.content[:150].strip()
            if len(res.case.content) > 150:
                content += '...'
            summaries.append(f"Case {i} (Relevance: {res.score:.2f}): {content}")
        return "\n".join(summaries)

    def _clean_case_ids_from_text(self, text: str) -> str:
        import re
        pattern = r'\[[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\]'
        cleaned_text = re.sub(pattern, '', text)
        cleaned_text = re.sub(r'\[\s*\]', '', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text

    def _clean_case_ids_from_response(self, response_dict: Dict) -> Dict:
        import re
        pattern = r'\[[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\]'
        
        cleaned_response = {}
        for key, value in response_dict.items():
            if isinstance(value, str):
                cleaned_value = re.sub(pattern, '', value)
                cleaned_value = re.sub(r'\[\s*\]', '', cleaned_value)
                cleaned_value = re.sub(r'\s+', ' ', cleaned_value).strip()
                cleaned_response[key] = cleaned_value
            elif isinstance(value, list):
                cleaned_list = []
                for item in value:
                    if isinstance(item, str):
                        cleaned_item = re.sub(pattern, '', item)
                        cleaned_item = re.sub(r'\[\s*\]', '', cleaned_item)
                        cleaned_item = re.sub(r'\s+', ' ', cleaned_item).strip()
                        cleaned_list.append(cleaned_item)
                    else:
                        cleaned_list.append(item)
                cleaned_response[key] = cleaned_list
            else:
                cleaned_response[key] = value
                
        return cleaned_response

    def _verify_insight(self, text: str, cases: List[RetrievalResult]) -> Dict:
        case_ids = [res.case.id for res in cases]
        confidence = 0.8 if cases else 0.0
        return {'confidence': round(confidence, 2), 'supporting_ids': case_ids}

    def _repair_json(self, text: str) -> Dict:
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            json_text = text[start:end]
            parsed_json = json.loads(json_text)
            return self._clean_case_ids_from_response(parsed_json)
        except:
            return {"error": "Invalid JSON response"}

    def get_avg_tokens(self) -> float:
        return sum(self.token_counts) / len(self.token_counts) if self.token_counts else 0
