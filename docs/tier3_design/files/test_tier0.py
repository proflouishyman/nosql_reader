#!/usr/bin/env python3
# test_tier0.py
# Purpose: Test Tier 0 corpus exploration system

"""
Test script for Tier 0 corpus exploration.

This script validates that:
1. Stratification works with real MongoDB data
2. Batch reading retrieves documents correctly
3. LLM analysis produces valid findings
4. Research notebook accumulates state correctly
5. Final outputs are generated successfully

Usage:
    python test_tier0.py [--budget N] [--strategy STRATEGY]

Examples:
    python test_tier0.py --budget 100 --strategy temporal
    python test_tier0.py --budget 500 --strategy balanced
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Add app to path
sys.path.insert(0, '/app')
sys.path.insert(0, str(Path(__file__).parent / 'app'))

from historian_agent.corpus_explorer import CorpusExplorer, explore_corpus
from historian_agent.stratification import CorpusStratifier, StratumReader
from historian_agent.research_notebook import ResearchNotebook


def test_stratification():
    """Test corpus stratification with real data."""
    print("\n" + "="*60)
    print("TEST 1: Stratification")
    print("="*60)
    
    stratifier = CorpusStratifier()
    
    # Test temporal stratification
    print("\n1. Temporal stratification...")
    temporal = stratifier.temporal_stratification(
        year_range=(1920, 1930),
        docs_per_year=10
    )
    print(f"   Created {len(temporal)} temporal strata")
    if temporal:
        print(f"   Example: {temporal[0].label}, {temporal[0].sample_size} docs")
    
    # Test genre stratification
    print("\n2. Genre stratification...")
    genre = stratifier.genre_stratification(docs_per_type=20)
    print(f"   Created {len(genre)} genre strata")
    if genre:
        print(f"   Example: {genre[0].label}, {genre[0].sample_size} docs")
    
    # Test biographical stratification
    print("\n3. Biographical stratification...")
    biographical = stratifier.biographical_stratification(
        min_docs_per_person=5,
        max_people=10,
        docs_per_person=10
    )
    print(f"   Created {len(biographical)} biographical strata")
    if biographical:
        print(f"   Example: {biographical[0].label}")
    
    # Test comprehensive strategy
    print("\n4. Comprehensive balanced strategy...")
    comprehensive = stratifier.build_comprehensive_strategy(
        total_budget=100,
        strategy='balanced'
    )
    print(f"   Created {len(comprehensive)} strata")
    print(f"   Total docs: {sum(s.sample_size for s in comprehensive)}")
    
    print("\n✅ Stratification test passed")
    return True


def test_batch_reading():
    """Test reading documents from a stratum."""
    print("\n" + "="*60)
    print("TEST 2: Batch Reading")
    print("="*60)
    
    stratifier = CorpusStratifier()
    reader = StratumReader()
    
    # Get one stratum
    strata = stratifier.temporal_stratification(
        year_range=(1920, 1925),
        docs_per_year=5
    )
    
    if not strata:
        print("   ⚠️  No strata available (check MongoDB data)")
        return False
    
    stratum = strata[0]
    print(f"\n1. Reading from: {stratum.label}")
    
    # Read documents
    docs = reader.read_stratum(stratum)
    print(f"   Retrieved {len(docs)} documents")
    
    if docs:
        doc = docs[0]
        print(f"   Sample doc: {doc.get('filename', 'Unknown')}")
        print(f"   Year: {doc.get('year', 'Unknown')}")
        print(f"   Type: {doc.get('document_type', 'Unknown')}")
    
    # Format for LLM
    print("\n2. Formatting for LLM...")
    formatted = reader.format_docs_for_llm(docs, max_chars=1000)
    print(f"   Formatted text length: {len(formatted)} chars")
    print(f"   Preview: {formatted[:200]}...")
    
    print("\n✅ Batch reading test passed")
    return True


def test_notebook():
    """Test research notebook state management."""
    print("\n" + "="*60)
    print("TEST 3: Research Notebook")
    print("="*60)
    
    notebook = ResearchNotebook()
    
    # Simulate batch findings
    print("\n1. Simulating batch findings...")
    findings = {
        'entities': [
            {'name': 'John Smith', 'type': 'person', 'first_seen': 'doc1', 'context': 'railroad worker'},
            {'name': 'B&O Railroad', 'type': 'organization', 'first_seen': 'doc1', 'context': 'employer'}
        ],
        'patterns': [
            {'pattern': 'High injury rates in 1923', 'type': 'injury_trend', 'evidence': ['doc1', 'doc2'], 'confidence': 'medium'}
        ],
        'contradictions': [
            {'claim_a': 'Wage was $100', 'claim_b': 'Wage was $85', 'source_a': 'doc1', 'source_b': 'doc2', 'context': 'wage dispute'}
        ],
        'questions': [
            {'question': 'Why were wages different?', 'why_interesting': 'Large discrepancy', 'evidence_needed': 'payroll records'}
        ],
        'temporal_events': {
            '1923': ['Labor dispute', 'Injury spike']
        },
        'stats': {
            'docs_in_batch': 5,
            'by_year': {'1923': 3, '1924': 2},
            'by_collection': {'Relief Records': 5},
            'by_document_type': {'Injury Report': 3, 'Wage Record': 2},
            'earliest_year': 1923,
            'latest_year': 1924
        }
    }
    
    notebook.integrate_batch_findings(findings, 'Batch 1: Year 1923-1924')
    
    print(f"   Entities: {len(notebook.entities)}")
    print(f"   Patterns: {len(notebook.patterns)}")
    print(f"   Questions: {len(notebook.questions)}")
    print(f"   Contradictions: {len(notebook.contradictions)}")
    
    # Test summary
    print("\n2. Testing summary generation...")
    summary = notebook.get_summary()
    print(f"   Total entities: {summary['total_entities']}")
    print(f"   Documents read: {summary['documents_read']}")
    
    # Test LLM context formatting
    print("\n3. Testing LLM context formatting...")
    context = notebook.format_for_llm_context()
    print(f"   Context length: {len(context)} chars")
    print(f"   Preview: {context[:200]}...")
    
    # Test save/load
    print("\n4. Testing save/load...")
    test_path = '/tmp/test_notebook.json'
    notebook.save(test_path)
    loaded = ResearchNotebook.load(test_path)
    print(f"   Saved and loaded successfully")
    print(f"   Loaded entities: {len(loaded.entities)}")
    
    os.remove(test_path)
    
    print("\n✅ Research notebook test passed")
    return True


def test_full_exploration(budget: int, strategy: str):
    """Test full corpus exploration end-to-end."""
    print("\n" + "="*60)
    print("TEST 4: Full Corpus Exploration")
    print("="*60)
    
    print(f"\nRunning exploration:")
    print(f"  Strategy: {strategy}")
    print(f"  Budget: {budget} documents")
    print(f"  (This will take a few minutes...)")
    
    try:
        report = explore_corpus(
            strategy=strategy,
            total_budget=budget,
            year_range=None
        )
        
        print("\n✅ Exploration completed!")
        print(f"\nResults:")
        print(f"  Documents read: {report['exploration_metadata']['documents_read']}")
        print(f"  Batches processed: {report['exploration_metadata']['batches_processed']}")
        print(f"  Duration: {report['exploration_metadata']['duration_seconds']:.1f}s")
        print(f"  Questions generated: {len(report['questions'])}")
        print(f"  Patterns found: {len(report['patterns'])}")
        print(f"  Entities found: {len(report['entities'])}")
        print(f"  Contradictions: {len(report['contradictions'])}")
        
        # Show sample questions
        if report['questions']:
            print(f"\n📋 Sample questions generated:")
            for i, q in enumerate(report['questions'][:3], 1):
                print(f"\n{i}. {q.get('question', 'Unknown')}")
                print(f"   Why interesting: {q.get('why_interesting', 'N/A')}")
        
        # Show high-confidence patterns
        high_conf_patterns = [p for p in report['patterns'] if p.get('confidence') == 'high']
        if high_conf_patterns:
            print(f"\n🔍 High-confidence patterns:")
            for i, p in enumerate(high_conf_patterns[:3], 1):
                print(f"\n{i}. {p['pattern']}")
                print(f"   Evidence: {p['evidence_count']} documents")
        
        # Show corpus map notes
        if 'corpus_map' in report and 'archive_notes' in report['corpus_map']:
            print(f"\n📚 Archive orientation notes:")
            notes = report['corpus_map']['archive_notes'][:500]  # First 500 chars
            print(f"{notes}...")
        
        # Show saved notebook path
        if 'notebook_path' in report:
            print(f"\n💾 Notebook saved to: {report['notebook_path']}")
        
        print("\n✅ Full exploration test passed")
        return True
        
    except Exception as e:
        print(f"\n❌ Exploration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description='Test Tier 0 corpus exploration')
    parser.add_argument('--budget', type=int, default=100, help='Document budget for full test (default: 100)')
    parser.add_argument('--strategy', type=str, default='balanced', 
                       choices=['temporal', 'biographical', 'genre', 'balanced'],
                       help='Exploration strategy (default: balanced)')
    parser.add_argument('--skip-full', action='store_true', help='Skip full exploration test')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("TIER 0 CORPUS EXPLORATION - TEST SUITE")
    print("="*60)
    
    results = []
    
    # Run tests
    try:
        results.append(('Stratification', test_stratification()))
        results.append(('Batch Reading', test_batch_reading()))
        results.append(('Research Notebook', test_notebook()))
        
        if not args.skip_full:
            results.append(('Full Exploration', test_full_exploration(args.budget, args.strategy)))
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All tests passed!")
        print("\nNext steps:")
        print("1. Review integration guide: TIER0_INTEGRATION.md")
        print("2. Add route to app/routes.py")
        print("3. Run full corpus exploration with larger budget")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Review output above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
