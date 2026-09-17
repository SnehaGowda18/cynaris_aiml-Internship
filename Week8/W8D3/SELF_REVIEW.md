\# W8D3 Self-Review



\## Project

Deployment: Haystack API on Railway



\## Completed Tasks



\- \[x] Built Haystack BM25 retrieval pipeline

\- \[x] Indexed 5 PDF documents

\- \[x] Created 10 retrieval questions

\- \[x] Evaluated BM25 retrieval quality

\- \[x] Implemented Dense Retrieval comparison

\- \[x] Compared BM25 and Dense Retrieval precision

\- \[x] Created FastAPI application

\- \[x] Added `/health` endpoint

\- \[x] Added `/retrieve` endpoint

\- \[x] Created Dockerfile

\- \[x] Built Docker image

\- \[x] Tested API using Docker



\## Docker



Image:



&#x20;   haystack-w8d3



Run:



&#x20;   docker run --rm -p 8000:8000 haystack-w8d3



Health test:



&#x20;   curl http://localhost:8000/health



Expected response:



&#x20;   {

&#x20;     "status": "healthy",

&#x20;     "retrieval\_method": "BM25",

&#x20;     "documents": 10

&#x20;   }



\## Retrieval



BM25 was used as the baseline retrieval method.



Dense Retrieval was implemented to compare semantic retrieval

against keyword-based BM25 retrieval.



Both approaches were evaluated using the same 10 questions

and the same top-k value.



\## Challenges



\- Haystack dependency compatibility

\- Missing NLTK dependency

\- Docker Desktop engine issues

\- Port 8000 conflicts

\- PowerShell JSON quoting issues



\## Learning



BM25 is effective for keyword matching, while Dense Retrieval

can retrieve semantically similar content even when the exact

query words are not present.



\## Self-Review



\- Code Quality: Completed

\- Git Workflow: Completed

\- Docker Testing: Completed

\- API Testing: Completed

\- Evaluation Evidence: Completed

\- Documentation: Completed

\- Railway Deployment: Pending

\- Pull Request: Pending



\## Future Improvements



\- Add automated precision and recall evaluation

\- Improve the evaluation dataset

\- Optimize Docker image size

\- Add production monitoring

\- Add automated CI/CD deployment

