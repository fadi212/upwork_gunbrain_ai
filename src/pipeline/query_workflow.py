from typing import Union
from collections import defaultdict
from llama_index.core.response_synthesizers import (
    ResponseMode,
    get_response_synthesizer,
)
from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
    Event,
)
from llama_index.llms.deepseek import DeepSeek
from llama_index.llms.gemini import Gemini
from llama_index.llms.anthropic import Anthropic
from llama_index.core.schema import NodeWithScore
from src.pipeline.prompts.internal.gemini_general import Gemini_QA_TEMPLATE_INTERNAL
from src.pipeline.prompts.internal.claude_general import Claude_QA_TEMPLATE_INTERNAL
from src.pipeline.prompts.internal.general import GENERAL_QA_TEMPLATE_INTERNAL

from src.pipeline.prompts.internal.audit_info import AUDIT_PROMPT_INTERNAL
from src.pipeline.prompts.internal.serial_lookup import SERIAL_LOOKUP_INTERNAL
from src.pipeline.prompts.external.general import (
    GENERAL_QA_TEMPLATE_EXTERNAL,
    REFINE_GENERAL_QA_TEMPLATE_EXTERNAL,
)
from src.pipeline.prompts.external.audit_info import AUDIT_PROMPT_EXTERNAL
from src.pipeline.prompts.external.serial_lookup import SERIAL_LOOKUP_EXTERNAL
from llama_index.core.prompts import PromptTemplate
from src.utils.utils import logging, format_chat_history
from src.utils.token_counter import count_tokens
# chat memory
from src.pipeline.memory import session_chat_memory
from llama_index.core.base.llms.types import ChatMessage, MessageRole
import os
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
DeepSeek_API_KEY = os.environ["DeepSeek_API_KEY"]
class RetrieverEvent(Event):
    nodes: list[NodeWithScore]

class CitationQueryEngineWorkflow(Workflow):
    def __init__(self, timeout=500):
        super().__init__(timeout=timeout)

    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> Union[RetrieverEvent, None]:
        query = ev.get("query")
        query_type = ev.get("query_type")
        prompt_type = ev.get("other_args").get("prompt_type")
        session_id = ev.get("session_id")
        logging.info(f"Query: {query} (Session: {session_id})")

        # Retrieve chat history from the session memory (as a list)
        memory = session_chat_memory[session_id]
        messages = memory.get_all()  # returns list of ChatMessage
        history = format_chat_history(messages)
        
        # Store values in context
        await ctx.set("query", query)
        await ctx.set("query_type", query_type)
        await ctx.set("prompt_type", prompt_type)
        await ctx.set("session_id", session_id)
        await ctx.set("history", history)

        selected_model = ev.get("other_args").get("selected_model", "gpt-4o-mini")
        await ctx.set("selected_model", selected_model)

        if ev.index is None:
            raise ValueError("Index is empty, load documents before querying!")

        retriever = ev.index.as_retriever(similarity_top_k=5)
        nodes = retriever.retrieve(query)
        logging.info(f"Retrieved {len(nodes)} nodes.")

        # Add the user's query to the chat memory
        memory.put(ChatMessage(role=MessageRole.USER, content=query))
        return RetrieverEvent(nodes=nodes)

    @step
    async def synthesize(self, ctx: Context, ev: RetrieverEvent) -> StopEvent:
        selected_model = await ctx.get("selected_model")
        if selected_model == "gpt-4o-mini":
            llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
        elif selected_model == "gemini-2.0-flash-001":
            llm = Gemini(model="models/gemini-1.5-flash-002")
        elif selected_model == "claude-3-7-sonnet-latest":
            llm = Anthropic(model="claude-3-7-sonnet-latest")
        elif selected_model == "claude-3-7-sonnet-20250219":
            llm = Anthropic(model="claude-3-7-sonnet-20250219")
        elif selected_model == "claude-3-5-sonnet-latest":
            llm = Anthropic(model="claude-3-5-sonnet-latest")
        elif selected_model == "deepseek":
            llm = DeepSeek(model='deepseek-chat', api_key=DeepSeek_API_KEY)
        else:
            raise ValueError(f"Unsupported model selected: {selected_model}")

        query = await ctx.get("query")
        session_id = await ctx.get("session_id")
        query_type = await ctx.get("query_type")
        prompt_type = await ctx.get("prompt_type")
        context_str = self.build_context_str(ev.nodes)

        full_qa_template = ""
        if query_type == "internal":
            if prompt_type == "General":
                # If model is Gemini, use the gemini_general prompt
                if selected_model == "gemini-2.0-flash-001":
                    full_qa_template = Gemini_QA_TEMPLATE_INTERNAL.format(context_str=context_str, query_str=query)
                elif selected_model == "claude-3-5-sonnet-latest":
                    full_qa_template = Claude_QA_TEMPLATE_INTERNAL.format(context_str=context_str, query_str=query)
                elif selected_model == "claude-3-7-sonnet-latest":
                    full_qa_template = Claude_QA_TEMPLATE_INTERNAL.format(context_str=context_str, query_str=query)
                else:
                    # Otherwise, use the GPT-4o-mini internal general prompt
                    full_qa_template = GENERAL_QA_TEMPLATE_INTERNAL.format(context_str=context_str, query_str=query)
                refine_template = None
                refine_template = None
            elif prompt_type == "Information Audit":
                full_qa_template = AUDIT_PROMPT_INTERNAL.format(context_str=context_str, query_str=query)
                refine_template = None
            elif prompt_type == "Serial Number Lookup":
                full_qa_template = SERIAL_LOOKUP_INTERNAL.format(context_str=context_str, query_str=query)
                refine_template = None
            else:
                raise ValueError("The prompt type is not valid")
        elif query_type == "external":
            if prompt_type == "General":
                full_qa_template = GENERAL_QA_TEMPLATE_EXTERNAL.format(context_str=context_str, query_str=query)
                refine_template = REFINE_GENERAL_QA_TEMPLATE_EXTERNAL
            elif prompt_type == "Information Audit":
                full_qa_template = AUDIT_PROMPT_EXTERNAL.format(context_str=context_str, query_str=query)
                refine_template = None
            elif prompt_type == "Serial Number Lookup":
                full_qa_template = SERIAL_LOOKUP_EXTERNAL.format(context_str=context_str, query_str=query)
                refine_template = None
            else:
                raise ValueError("The prompt type is not valid")
        else:
            raise ValueError("The query type is not valid")

        qa_prompt_template = PromptTemplate(full_qa_template)
        refine_prompt_template = PromptTemplate(refine_template.format(
            context_msg=context_str, query_str=query, existing_answer="{existing_answer}"
        )) if refine_template else None

        prompt_tokens = count_tokens(full_qa_template)
        logging.debug(f"Manual Prompt Token Count: {prompt_tokens}")

        synthesizer = get_response_synthesizer(
            llm=llm,
            text_qa_template=qa_prompt_template,
            refine_template=refine_prompt_template,
            response_mode=ResponseMode.COMPACT,
            use_async=True,
        )
        response = await synthesizer.asynthesize(query=query, nodes=ev.nodes)

        try:
            completion_text = response.response
        except AttributeError:
            try:
                completion_text = response.generations[0].text
            except AttributeError:
                completion_text = str(response)

        logging.debug(f"LLM Completion Text: {completion_text}")

        completion_tokens = count_tokens(completion_text)
        total_tokens = prompt_tokens + completion_tokens

        logging.debug(f"Manual Completion Token Count: {completion_tokens}")
        logging.debug(f"Manual Total Token Count: {total_tokens}")

        # Add the assistant's response to chat memory.
        memory = session_chat_memory[session_id]
        memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=completion_text))

        # Include the formatted chat history in the output.
        chat_history = format_chat_history(memory.get_all())

        return StopEvent(result={
            "response": completion_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "source_nodes": ev.nodes,
            "chat_history": chat_history,
            "model_used": selected_model
        })

    def build_context_str(self, nodes):
        source_dict = defaultdict(list)
        for idx, node in enumerate(nodes):
            source_dict[idx + 1].append(node.node.text)
        context_str = ""
        for source_number in sorted(source_dict.keys()):
            context_str += f"Source {source_number}:\n" + "\n".join(source_dict[source_number]) + "\n\n"
        return context_str
