#!/usr/bin/env python3
"""
Model Performance Tester for Historical Document RAG
Tests speed and quality across different Ollama models
"""

import time
import json
import requests
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# Test configuration
OLLAMA_BASE_URL = "http://localhost:11434"
TEST_QUERY = "What kinds of injuries did firemen get?"
MODELS_TO_TEST = [
    "gpt-oss:20b",
    "qwen2.5:32b", 
    "llama3.2:11b",
    "mistral:7b"
]

class ModelTester:
    def __init__(self, rag_endpoint: str = "http://localhost:5001/api/query"):
        self.rag_endpoint = rag_endpoint
        self.results = []
    
    def test_model(self, model: str, query: str, top_k: int = 50) -> Dict:
        """Test a single model with timing and quality metrics"""
        print(f"\n🧪 Testing {model}...")
        
        start = time.time()
        try:
            response = requests.post(
                self.rag_endpoint,
                json={
                    "query": query,
                    "top_k": top_k,
                    "model": model
                },
                timeout=180
            )
            elapsed = time.time() - start
            
            if response.status_code != 200:
                return {
                    "model": model,
                    "status": "failed",
                    "error": response.text,
                    "time": elapsed
                }
            
            data = response.json()
            answer = data.get("answer", "")
            
            # Quality metrics
            quality = {
                "length": len(answer),
                "sources_cited": len(data.get("sources", [])),
                "confidence": self._estimate_confidence(answer),
                "specificity": self._count_specifics(answer)
            }
            
            return {
                "model": model,
                "status": "success",
                "time": elapsed,
                "answer_length": quality["length"],
                "sources": quality["sources_cited"],
                "confidence": quality["confidence"],
                "specifics": quality["specificity"],
                "answer": answer[:500] + "..." if len(answer) > 500 else answer
            }
            
        except Exception as e:
            elapsed = time.time() - start
            return {
                "model": model,
                "status": "error",
                "error": str(e),
                "time": elapsed
            }
    
    def _estimate_confidence(self, answer: str) -> str:
        """Estimate answer confidence from hedging language"""
        hedges = ["might", "possibly", "unclear", "uncertain", "limited information"]
        if any(h in answer.lower() for h in hedges):
            return "low"
        return "medium" if len(answer) < 500 else "high"
    
    def _count_specifics(self, answer: str) -> int:
        """Count specific details (dates, names, numbers)"""
        import re
        dates = len(re.findall(r'\b\d{4}\b', answer))
        numbers = len(re.findall(r'\b\d+\b', answer))
        capitals = len(re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', answer))
        return dates + numbers + capitals
    
    def run_tests(self, query: str = TEST_QUERY) -> List[Dict]:
        """Run tests across all models"""
        print(f"🔬 Testing {len(MODELS_TO_TEST)} models")
        print(f"Query: {query}\n")
        
        for model in MODELS_TO_TEST:
            result = self.test_model(model, query)
            self.results.append(result)
            
            # Print immediate feedback
            if result["status"] == "success":
                print(f"✅ {model}: {result['time']:.1f}s | {result['sources']} sources | {result['specifics']} specifics")
            else:
                print(f"❌ {model}: {result['status']} - {result.get('error', 'unknown')}")
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate comparison report"""
        report = [
            "\n" + "="*80,
            "MODEL PERFORMANCE COMPARISON",
            "="*80,
            f"Query: {TEST_QUERY}",
            f"Timestamp: {datetime.now().isoformat()}",
            "\n"
        ]
        
        # Sort by time
        successful = [r for r in self.results if r["status"] == "success"]
        successful.sort(key=lambda x: x["time"])
        
        if not successful:
            report.append("⚠️  No successful tests")
            return "\n".join(report)
        
        # Performance table
        report.append("SPEED RANKING:")
        report.append("-" * 80)
        for i, r in enumerate(successful, 1):
            report.append(
                f"{i}. {r['model']:20s} | "
                f"{r['time']:6.1f}s | "
                f"{r['sources']:2d} sources | "
                f"{r['specifics']:3d} details | "
                f"conf: {r['confidence']}"
            )
        
        # Quality analysis
        report.append("\nQUALITY METRICS:")
        report.append("-" * 80)
        best_detail = max(successful, key=lambda x: x["specifics"])
        fastest = min(successful, key=lambda x: x["time"])
        most_sources = max(successful, key=lambda x: x["sources"])
        
        report.append(f"🏆 Most Detailed:    {best_detail['model']} ({best_detail['specifics']} specifics)")
        report.append(f"⚡ Fastest:          {fastest['model']} ({fastest['time']:.1f}s)")
        report.append(f"📚 Most Sources:     {most_sources['model']} ({most_sources['sources']} documents)")
        
        # Recommendation
        report.append("\nRECOMMENDATION:")
        report.append("-" * 80)
        
        # Score: balance speed and quality
        for r in successful:
            r["score"] = r["specifics"] / max(r["time"], 1)
        
        best = max(successful, key=lambda x: x["score"])
        report.append(f"✨ Best Overall: {best['model']}")
        report.append(f"   (Quality/Speed ratio: {best['score']:.2f})")
        
        return "\n".join(report)
    
    def save_results(self, filepath: str = "model_test_results.json"):
        """Save detailed results to JSON"""
        output = {
            "timestamp": datetime.now().isoformat(),
            "query": TEST_QUERY,
            "models_tested": len(MODELS_TO_TEST),
            "results": self.results
        }
        
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Results saved to {filepath}")

if __name__ == "__main__":
    tester = ModelTester()
    tester.run_tests()
    print(tester.generate_report())
    tester.save_results()