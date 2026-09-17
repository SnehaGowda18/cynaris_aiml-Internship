from crewai.tools import BaseTool
import requests
from bs4 import BeautifulSoup


class WebSearchTool(BaseTool):
    name: str = "Web Search"
    description: str = (
        "Search the web for current information about a topic "
        "and return relevant search results."
    )

    def _run(self, query: str) -> str:
        try:
            url = "https://www.google.com/search"

            params = {
                "q": query
            }

            headers = {
                "User-Agent": "Mozilla/5.0"
            }

            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=10
            )

            response.raise_for_status()

            soup = BeautifulSoup(
                response.text,
                "html.parser"
            )

            results = []

            for result in soup.select("div"):
                text = result.get_text(" ", strip=True)

                if len(text) > 80:
                    results.append(text)

            if not results:
                return "No useful web search results found."

            # Remove duplicates
            unique_results = []

            for result in results:
                if result not in unique_results:
                    unique_results.append(result)

            return "\n".join(unique_results[:10])

        except Exception as e:
            return f"Web search failed: {str(e)}"