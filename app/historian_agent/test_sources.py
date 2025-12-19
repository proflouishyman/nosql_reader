#!/usr/bin/env python3
from .adversarial_rag import AdversarialRAGHandler

handler = AdversarialRAGHandler()
try:
    # Use a simple string for the query
    ans, lat, sources = handler.process_query('test')
    
    print(f"\nLATENCY: {lat:.2f}s")
    print('Sources type:', type(sources))
    print('Sources:', sources)
finally:
    handler.close()