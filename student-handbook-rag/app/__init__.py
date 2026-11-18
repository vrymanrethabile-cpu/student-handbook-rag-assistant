"""Student Handbook RAG Assistant - application package.

Importing this package injects the Windows certificate store into SSL
verification (via ``truststore``). Some networks inspect HTTPS traffic,
which breaks the default certifi bundle; truststore makes outbound calls
to Hugging Face / LLM providers work without disabling verification.
"""

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    # truststore is optional; without it we fall back to certifi defaults.
    pass
