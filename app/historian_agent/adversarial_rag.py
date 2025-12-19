#!/usr/bin/env python3
import sys, os, time, json, re
import requests
from .rag_query_handler import RAGQueryHandler
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
DEBUG = os.environ.get("DEBUG_MODE", "0") == "1"
VERIFIER_MODEL = os.environ.get("VERIFIER_MODEL", os.environ.get("LLM_MODEL", "gpt-oss:20b"))
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

def debug_step(step_name: str, detail: str = "", icon: str = "⚡", level: str = "INFO"):
    if not DEBUG: return
    timestamp = time.strftime("%H:%M:%S")
    color = "⚠️" if level == "WARN" else "❌" if level == "ERROR" else icon
    sys.stderr.write(f"{color} [{timestamp}] [ADVERSARIAL] {step_name.upper()}\n")
    if detail: sys.stderr.write(f"   └─ {detail}\n")
    sys.stderr.flush()

class AdversarialRAGHandler:
    def __init__(self):
        debug_step("Init", "Initializing Adversarial Handler...", icon="🛡️")
        self.rag_handler = RAGQueryHandler()

    def verify_citations(self, question, answer, sources_text):
        """
        The 'Judge' Step:
        Asks the LLM to verify if the Answer is supported by the Source Text.
        """
        debug_step("Verification", "Cross-checking claims against source text...", icon="⚖️")
        
        SYSTEM_PROMPT = """
            You are a strict fact-checking judge.

            Your task:
            1. Break the ANSWER into discrete factual claims.
            2. Verify each claim using ONLY the SOURCE TEXT.
            3. A claim is supported ONLY if it is explicitly stated in the SOURCE TEXT.
            - Do NOT use background knowledge.
            - Do NOT infer or assume.
            4. If ANY factual claim is unsupported, mark the answer as inaccurate.

            Scoring:
            - citation_score = percentage of claims fully supported by the source.
            - 100 = all claims supported
            - 0 = no claims supported

            Output rules:
            - Respond with ONLY a valid JSON object. first char must be { and last char must be }
            - Use EXACTLY the schema provided.
            - Do NOT include markdown, commentary, or extra keys.
            """


        USER_PROMPT = f"""
            QUESTION:
            {question}

            ANSWER TO VERIFY:
            {answer}

            SOURCE TEXT:
            {sources_text[:60000]}

            Required JSON output format:
            {{
            "is_accurate": boolean,
            "citation_score": number,
            "reasoning": "List each unsupported or partially supported claim, or state that all claims are supported."
            }}
            """
        JSON_SCHEMA = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "is_accurate": {"type": "boolean"},
                "citation_score": {"type": "number"},
                "reasoning": {"type": "string"}
            },
            "required": ["is_accurate", "citation_score", "reasoning"]
        }

        SYSTEM_PROMPT = SYSTEM_PROMPT.strip()
        USER_PROMPT = USER_PROMPT.strip()
        

        try:
            resp = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json={
                "model": VERIFIER_MODEL,
                "system": SYSTEM_PROMPT,
                "prompt": USER_PROMPT,
                "stream": False,
                "format": JSON_SCHEMA,
                "options": {"temperature": 0.0}
            })
            
            raw_response = resp.json().get("response", "")
            
            # --- DEBUGGING: PRINT THE RAW OUTPUT ---
            # This will show us exactly what the LLM sent back
            debug_step("Raw Verdict", f"LLM Output: {raw_response[:500]}...", icon="📝")
            
            if not raw_response or not raw_response.strip():
                raise ValueError("Empty response from Verifier LLM")

            # --- REGEX PARSING ---
            # Finds the first '{' and the last '}' to ignore "Here is your JSON..." chatter
            json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group(0))
            else:
                return json.loads(raw_response)
                
        except Exception as e:
            debug_step("Verification Error", str(e), level="WARN")
            return {
                "is_accurate": False, 
                "citation_score": 0, 
                "reasoning": f"Judge Failed: {str(e)} | Raw: {raw_response[:100] if 'raw_response' in locals() else 'None'}"
            }

    def process_query(self, question: str):
        start = time.time()
        
        # 1. Generate Answer (Standard RAG)
        ans, metrics = self.rag_handler.process_query(question, label="ADVERSARIAL_GEN")
        
        # 2. Extract Context (What did the Generator see?)
        verify_text = metrics.get('context', "")
        
        # Fallback if 'context' wasn't returned by handler
        if not verify_text:
            source_ids = list(metrics.get('sources', {}).values())
            if source_ids:
                # Re-fetch top 3 documents text for the Judge
                verify_text, _, _ = self.rag_handler.get_full_document_text(source_ids[:3])
            else:
                verify_text = "No sources found."

        # 3. Verify (The Trial)
        verdict = self.verify_citations(question, ans, verify_text)
        
        score = verdict.get('citation_score', 0)
        reasoning = verdict.get('reasoning', 'No reasoning provided.')
        
        debug_step("Verdict", f"Score: {score}/100 | {reasoning}", icon="👩‍⚖️")
        
        # 4. Expose Reasoning
        if score < 90:
            separator = "\n\n" + "─"*40 + "\n"
            ans += f"{separator}🛡️ **VERIFICATION REPORT (Score: {score}/100)**\n"
            ans += f"**Judge's Reasoning:** {reasoning}\n"
            
        metrics['verification'] = verdict
        return ans, time.time() - start, metrics.get('sources', {})

    def close(self):
        self.rag_handler.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python adversarial_rag.py 'question'")
    
    # CLI Test Runner
    handler = AdversarialRAGHandler()
    try:
        query = sys.argv[1]
        ans, lat, src = handler.process_query(query)
        
        print("\n" + "═"*60)
        print(f"🤖 ADVERSARIAL RESPONSE ({lat:.2f}s)")
        print("─"*60)
        print(ans)
        print("═"*60)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        handler.close()