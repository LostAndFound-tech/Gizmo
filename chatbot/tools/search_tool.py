"""
tools/search_tool.py
Agent-facing web search tool.

Wraps epistemic_synthesis.research() — the full pipeline of
search → fetch 3 sources → cross-reference → truth-weight → ingest.

The agent calls this when it needs current or external information.
Results are ingested into 'web_knowledge' RAG collection automatically
so future queries can retrieve them without re-searching.

Args:
  query (str)      — what to search for
  session_id (str) — current session (for context attribution)
"""

from tools.base_tool import BaseTool, ToolResult


class SearchTool(BaseTool):
    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web for current or external information. "
            "EXPLICIT TRIGGERS: 'look this up', 'search for', 'find out', "
            "'what's the latest on', 'check online', 'look it up', "
            "'what's the current', 'google that', 'can you find'. "
            "IMPLICIT TRIGGERS (routed here automatically): queries about "
            "current events, recent news, prices, live data, people in the news, "
            "or anything time-sensitive that local memory won't have. "
            "Fetches 3 sources, cross-references them, returns a synthesis that "
            "distinguishes what's well-established from what's contested. "
            "Results saved to memory for future reference. "
            "If the query is ambiguous, asks for clarification before searching. "
            "Args: query (str) — what to search for."
        )

    async def run(
        self,
        query: str = "",
        session_id: str = "",
        **kwargs,
    ) -> ToolResult:
        if not query.strip():
            return ToolResult(
                success=False,
                output="No search query provided.",
            )

        try:
            from tools.epistemic_synthesis import research, format_for_response
            from core.llm import llm

            # Get current host for attribution in RAG
            current_host = kwargs.get("current_host", "")

            result = await research(
                query=query.strip(),
                llm=llm,
                n_sources=3,
                ingest=True,
                current_host=current_host,
            )

            # If disambiguation needed, surface the question
            if result.needs_clarification:
                return ToolResult(
                    success=True,
                    output=(
                        f"Before I search, I need to clarify: {result.clarification_question} "
                        f"Ask the user this question before proceeding."
                    ),
                    data={"needs_clarification": True, "question": result.clarification_question},
                )

            formatted = format_for_response(result)

            return ToolResult(
                success=True,
                output=formatted,
                data={
                    "query": result.query,
                    "consensus_count": len(result.consensus),
                    "contested_count": len(result.contested),
                    "sources": result.sources,
                    "ingested": result.ingested,
                },
            )

        except Exception as e:
            print(f"[SearchTool] Failed: {e}")
            return ToolResult(
                success=False,
                output=f"Search failed: {e}. This may be a network or API key issue.",
            )