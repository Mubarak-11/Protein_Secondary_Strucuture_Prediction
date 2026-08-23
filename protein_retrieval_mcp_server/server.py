""" Custom MCP server for exposing protein retrieval tools. """

import logging
import uuid

from mcp.server.fastmcp import FastMCP
from typing import Annotated, Any
from pydantic import Field

from protein_retrieval.config import MIN_TOP_K, MAX_TOP_K

#MCP server
mcp = FastMCP("Protein Retrieval MCP server")

def _success(data: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "data": data}


def _error(exc: Exception) -> dict[str, Any]:
    error_id = str(uuid.uuid4())
    logging.exception("Protein retrieval MCP tool failed: %s", error_id)

    return {
        "ok": False,
        "error": {
            "code": "protein_retrieval_tool_error",
            "message": "Protein retrieval tool failed.",
            "error_id": error_id,
        },
    }


def _clamp_top_k(top_k: int) -> int:
    return max(MIN_TOP_K, min(top_k, MAX_TOP_K))


#----MCP Tools----
#Tool #1: Semantic search proteins
@mcp.tool()
def semantic_search_proteins(
    query: Annotated[str, Field(min_length=1, description="Protein search query")],
    top_k: Annotated[int, Field(ge=MIN_TOP_K, le=MAX_TOP_K)] = 5,
) -> dict[str, Any]:
    """ Search proteins by semantic meaning using BGE embeddings and pgvector. """

    try:

        from protein_retrieval.service import semantic_search_proteins as run_semantic_search

        result = run_semantic_search(
            query = query,
            top_k= _clamp_top_k(top_k)
        )
        return _success(result)

    except Exception as exc:
        return _error(exc)


#Tool #2: Keyword Search proteins
@mcp.tool()
def keyword_search_proteins(
    query: Annotated[str, Field(min_length=1, description="Protein search query")],
    top_k: Annotated[int, Field(ge=MIN_TOP_K, le=MAX_TOP_K)] = 5,
) -> dict[str, Any]:
    """ Search proteins  by exact lexical terms using PostgreSQL full-text search. """


    try:

        from protein_retrieval.service import keyword_search_proteins as run_keyword_search

        result = run_keyword_search(
            query=query,
            top_k= _clamp_top_k(top_k)
        )
        return _success(result)
        
    except Exception as exc:
        return _error(exc)


#Tool #3: Hybrid Search Proteins
@mcp.tool()
def hybrid_search_proteins(
    query: Annotated[str, Field(min_length=1, description="Protein search query")],
    top_k: Annotated[int, Field(ge=MIN_TOP_K, le=MAX_TOP_K)] = 5,
    candidate_k: Annotated[int, Field(ge=MIN_TOP_K, le=MAX_TOP_K)] = 20,
) -> dict[str, Any]:
    
    """ Search proteins using weighted hybrid retrieval.

    Combines semantic pgvector search with PostgreSQL full-text search using
    weighted reciprocal rank fusion. This is the default retrieval tool for
    most protein discovery questions. """


    try:

        from protein_retrieval.service import hybrid_search_proteins as run_hybrid_search

        result = run_hybrid_search(
            query= query,
            top_k= _clamp_top_k(top_k),
            candidate_k=candidate_k
        )
        return _success(result)

    except Exception as exc:
        return _error(exc)
    

if __name__=="__main__":
    mcp.run(transport="stdio")
