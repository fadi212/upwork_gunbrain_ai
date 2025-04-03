from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from src.utils.utils import index_name, pc, split_response_and_citations, logging
from src.pipeline.query_workflow import CitationQueryEngineWorkflow

async def query_index_async(query_type: str, query_text: str, **kwargs) -> dict:
    index_list_response = pc.list_indexes()
    available_indexes = [idx_info['name'] for idx_info in index_list_response.get('indexes', [])]
    logging.info(f"Available indexes before querying: {available_indexes}")

    if index_name not in available_indexes:
        return {"error": f"Index '{index_name}' does not exist in Pinecone."}
    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    embed_model = OpenAIEmbedding()
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
    session_id = kwargs.get("session_id", "default")
    workflow = CitationQueryEngineWorkflow(timeout=500)

    result = await workflow.run(
        query=query_text,
        index=index,
        query_type=query_type,
        session_id=session_id,
        other_args=kwargs
    )

    response = result.get("response", "")
    prompt_tokens = result.get("prompt_tokens", 0)
    completion_tokens = result.get("completion_tokens", 0)
    total_tokens = result.get("total_tokens", 0)

    logging.info(
        f"[QUERY] type={query_type}, prompt_tokens={prompt_tokens}, "
        f"completion_tokens={completion_tokens}, total_tokens={total_tokens}"
    )

    if query_type == 'external':
        return {
            "response": response,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "chat_history": result.get("chat_history"),
            "model_used": result.get("model_used")
        }

    source_nodes = result.get("source_nodes", [])
    citation_map = {}
    all_citations = []
    for idx, node_with_score in enumerate(source_nodes):
        citation_key = str(idx + 1)  # e.g., "1", "2"
        citation_map[citation_key] = node_with_score.node

        # Collect minimal metadata about all source nodes
        all_citations.append({
            'text': node_with_score.node.text,
            'type': node_with_score.metadata.get('type'),
            'author': node_with_score.metadata.get('author'),
            'date_published': node_with_score.metadata.get('published_date'),
            'issue_date': node_with_score.metadata.get('issue_date'),
            'manufacturer': node_with_score.metadata.get('manufacturer'),
            'origin_type': node_with_score.metadata.get('origin_type'),
            'bbox': node_with_score.metadata.get('bbox', []),
            'page_number': node_with_score.metadata.get('page_number'),
            'title': node_with_score.metadata.get('title'),
            's3_URL': node_with_score.metadata.get('s3_URL')
        })

    logging.info(f"All citations collected: {all_citations}")
    response_text, citation_numbers = split_response_and_citations(response)

    # Build a list of valid citation metadata
    metadata_list = []
    for citation_list in citation_numbers:
        for citation_num in citation_list:
            node = citation_map.get(str(citation_num))
            if node:
                # Valid citation found
                metadata = {
                    'citation': str(citation_num),
                    'type': node.metadata.get('type'),
                    'text': node.text,
                    'bbox': node.metadata.get('bbox', []),
                    'page_number': node.metadata.get('page_number'),
                    'title': node.metadata.get('title'),
                    'origin_type': node.metadata.get('origin_type'),
                    'manufacturer': node.metadata.get('manufacturer'),
                    'date_published': node.metadata.get('published_date'),
                    'author': node.metadata.get('author'),
                    'issue_date': node.metadata.get('issue_date'),
                    's3_URL': node.metadata.get('s3_URL')
                }
                metadata_list.append(metadata)
            else:
                # The model cited something that's not in citation_map
                metadata_list.append({
                    'citation': str(citation_num),
                    'error': 'Citation not found'
                })

    return {
        "response": response_text,
        "citations": metadata_list,       
        "all_citations": all_citations,    # All retrieved nodes
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "chat_history": result.get("chat_history"),
        "model_used": result.get("model_used")
    }

async def internal_query_index(prompt_type: str, query: str, session_id: str, selected_model: str = "gpt-4o-mini"):
    result = await query_index_async("internal", query, prompt_type=prompt_type, session_id=session_id, selected_model=selected_model)
    return (
        result["response"],
        result["citations"],
        result["prompt_tokens"],
        result["completion_tokens"],
        result["total_tokens"],
        result.get("chat_history"),
        result.get("model_used")
    )

async def external_query_index(prompt_type: str, query: str, session_id: str, selected_model: str = "gpt-4o-mini"):
    result = await query_index_async("external", query, prompt_type=prompt_type, session_id=session_id, selected_model=selected_model)
    return (
        result["response"],
        result.get("model_used"),
        result["prompt_tokens"],
        result["completion_tokens"],
        result["total_tokens"],
        result.get("chat_history")
    )
