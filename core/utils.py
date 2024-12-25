"""Utils."""

from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.gemini import Gemini
from llama_index.core.llms.llm import LLM
from llama_index.core.llms.utils import resolve_llm
from pydantic import BaseModel, Field
import os
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.agent import ReActAgent
from llama_index.core.agent.react.prompts import REACT_CHAT_SYSTEM_HEADER
from llama_index.core import (
    VectorStoreIndex,
    SummaryIndex,
    ServiceContext,
    Document,
)
from typing import List, cast, Optional
from llama_index.core import SimpleDirectoryReader
from llama_index.core.embeddings.utils import resolve_embed_model
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent.types import BaseAgent
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.agent.react.formatter import ReActChatFormatter
from llama_index.llms.openai.base import is_function_calling_model
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from core.builder_config import BUILDER_LLM
from typing import Dict, Tuple, Any
import streamlit as st

from llama_index.core.callbacks import CallbackManager, trace_method
from core.callback_manager import StreamlitFunctionsCallbackHandler
from llama_index.core.schema import ImageNode, NodeWithScore

### BETA: Multi-modal
from llama_index.core.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.indices.multi_modal.retriever import (
    MultiModalVectorIndexRetriever,
)
from llama_index.core.llms import ChatMessage
from llama_index.core.query_engine.multi_modal import SimpleMultiModalQueryEngine
from llama_index.core.chat_engine.types import (
    AGENT_CHAT_RESPONSE_TYPE,
    StreamingAgentChatResponse,
    AgentChatResponse,
)
from llama_index.core.llms import ChatResponse
from typing import Generator


class RAGParams(BaseModel):
    """RAG parameters.

    Parameters used to configure a RAG pipeline.

    """

    include_summarization: bool = Field(
        default=False,
        description=(
            "Whether to include summarization in the RAG pipeline. (only for GPT-4)"
        ),
    )
    top_k: int = Field(
        default=2, description="Number of documents to retrieve from vector store."
    )
    chunk_size: int = Field(default=1024, description="Chunk size for vector store.")
    embed_model: str = Field(
        default='gemini', description="Embedding model to use. Set to 'local' to avoid using OpenAI. (default is OpenAI)"
    )
    llm: str = Field(
        default="models/gemini-1.5-flash", description="LLM to use for summarization."
    )


# def _resolve_llm(llm_str: str) -> LLM:
#     """Resolve LLM."""
#     # TODO: make this less hardcoded with if-else statements
#     # see if there's a prefix
#     # - if there isn't, assume it's an OpenAI model
#     # - if there is, resolve it
#     tokens = llm_str.split(":")
#     if len(tokens) == 1 and tokens[0].startswith("gemini"):
#         os.environ["GOOGLE_API_KEY"] = st.secrets.gemini_key
#         llm = Gemini(model=tokens[0])
#     elif tokens[0] == "gemini":
#         os.environ["GOOGLE_API_KEY"] = st.secrets.gemini_key
#         llm = Gemini(model=tokens[1])
#     elif tokens[0] == "openai":
#         os.environ["OPENAI_API_KEY"] = st.secrets.openai_key
#         llm = OpenAI(model=tokens[1])
#     elif tokens[0] == "anthropic":
#         os.environ["ANTHROPIC_API_KEY"] = st.secrets.anthropic_key
#         llm = Anthropic(model=tokens[1])
#     else:
#         raise ValueError(f"LLM {llm_str} not recognized.")
#     return llm

def _resolve_llm(llm_str: str) -> LLM:
    tokens = llm_str.split(":")
    if tokens[0] == "gemini":
        os.environ["GOOGLE_API_KEY"] = st.secrets.gemini_key
        return Gemini(model=tokens[1])
    else:
        raise ValueError(f"LLM {llm_str} not recognized.")


def load_data(
    file_names: Optional[List[str]] = None,
    directory: Optional[str] = None,
    urls: Optional[List[str]] = None,
) -> List[Document]:
    """Load data."""
    file_names = file_names or []
    directory = directory or ""
    urls = urls or []

    # get number depending on whether specified
    num_specified = sum(1 for v in [file_names, urls, directory] if v)

    if num_specified == 0:
        raise ValueError("Must specify either file_names or urls or directory.")
    elif num_specified > 1:
        raise ValueError("Must specify only one of file_names or urls or directory.")
    elif file_names:
        reader = SimpleDirectoryReader(input_files=file_names)
        docs = reader.load_data()
    elif directory:
        reader = SimpleDirectoryReader(input_dir=directory)
        docs = reader.load_data()
    elif urls:
        from llama_hub.web.simple_web.base import SimpleWebPageReader

        # use simple web page reader from llamahub
        loader = SimpleWebPageReader()
        docs = loader.load_data(urls=urls)
    else:
        raise ValueError("Must specify either file_names or urls or directory.")

    return docs


def load_agent(
    tools: List,
    llm: LLM,
    system_prompt: str,
    extra_kwargs: Optional[Dict] = None,
    **kwargs: Any,
) -> BaseChatEngine:
    """Load agent."""
    extra_kwargs = extra_kwargs or {}
    if isinstance(llm, OpenAI) and is_function_calling_model(llm.model):
        # TODO: use default msg handler
        # TODO: separate this from agent_utils.py...
        def _msg_handler(msg: str) -> None:
            """Message handler."""
            st.info(msg)
            st.session_state.agent_messages.append(
                {"role": "assistant", "content": msg, "msg_type": "info"}
            )

        # add streamlit callbacks (to inject events)
        handler = StreamlitFunctionsCallbackHandler(_msg_handler)
        callback_manager = CallbackManager([handler])
        # get OpenAI Agent
        agent: BaseChatEngine = OpenAIAgent.from_tools(
            tools=tools,
            llm=llm,
            system_prompt=system_prompt,
            **kwargs,
            callback_manager=callback_manager,
        )
    else:
        if "vector_index" not in extra_kwargs:
            raise ValueError(
                "Must pass in vector index for CondensePlusContextChatEngine."
            )
        vector_index = cast(VectorStoreIndex, extra_kwargs["vector_index"])
        rag_params = cast(RAGParams, extra_kwargs["rag_params"])
        # use condense + context chat engine
        agent = CondensePlusContextChatEngine.from_defaults(
            vector_index.as_retriever(similarity_top_k=rag_params.top_k),
        )

    return agent


def load_meta_agent(
    tools: List,
    llm: LLM,
    system_prompt: str,
    extra_kwargs: Optional[Dict] = None,
    **kwargs: Any,
) -> BaseAgent:
    """Load meta agent.

    TODO: consolidate with load_agent.

    The meta-agent *has* to perform tool-use.

    """
    extra_kwargs = extra_kwargs or {}
    if isinstance(llm, OpenAI) and is_function_calling_model(llm.model):
        # get OpenAI Agent

        agent: BaseAgent = OpenAIAgent.from_tools(
            tools=tools,
            llm=llm,
            system_prompt=system_prompt,
            **kwargs,
        )
    else:
        agent = ReActAgent.from_tools(
            tools=tools,
            llm=llm,
            react_chat_formatter=ReActChatFormatter(
                system_header=system_prompt + "\n" + REACT_CHAT_SYSTEM_HEADER,
            ),
            **kwargs,
        )

    return agent


def construct_agent(
    system_prompt: str,
    rag_params: RAGParams,
    docs: List[Document],
    vector_index: Optional[VectorStoreIndex] = None,
    additional_tools: Optional[List] = None,
) -> Tuple[BaseChatEngine, Dict]:
    """Construct agent from docs / parameters / indices."""
    extra_info = {}
    additional_tools = additional_tools or []

    # first resolve llm and embedding model
    embed_model = resolve_embed_model(rag_params.embed_model)
    
    # llm = resolve_llm(rag_params.llm)
    # TODO: use OpenAI for now
    # llm = OpenAI(model=rag_params.llm)
    # llm = Gemini(model="models/gemini-1.5-flash")
    llm = _resolve_llm(rag_params.llm)

    # first let's index the data with the right parameters
    # service_context = ServiceContext.from_defaults(
    #     chunk_size=rag_params.chunk_size,
    #     llm=llm,
    #     embed_model=embed_model,
    # )
    # Đổi thành dòng dưới
    from llama_index.core.settings import Settings

    # settings = Settings(
    #     chunk_size=rag_params.chunk_size,
    #     llm=llm,
    #     embed_model=embed_model,
    # )

    print(f"embed_model: {embed_model}")  # Debug
    print(f"Settings.embed_model: {Settings.embed_model}")  # Debug
    print(f"rag_params.embed_model: {rag_params.embed_model}")  # Debug
    print(f"rag_params.llm: {rag_params.llm}")  # Debug
    print(f"Settings.llm: {Settings.llm}")  # Debug

    if vector_index is None:
        # vector_index = VectorStoreIndex.from_documents(
        #     docs, service_context=settings
        # )
        vector_index = VectorStoreIndex.from_documents(
            docs,
            chunk_size=Settings.chunk_size,
            llm=Settings.llm,
            embed_model=Settings.embed_model,
        )
    else:
        pass

    extra_info["vector_index"] = vector_index

    vector_query_engine = vector_index.as_query_engine(
        similarity_top_k=rag_params.top_k
    )
    all_tools = []
    vector_tool = QueryEngineTool(
        query_engine=vector_query_engine,
        metadata=ToolMetadata(
            name="vector_tool",
            description=("Use this tool to answer any user question over any data."),
        ),
    )
    all_tools.append(vector_tool)
    if rag_params.include_summarization:
        summary_index = SummaryIndex.from_documents(
            docs,
            chunk_size=Settings.chunk_size,
            llm=Settings.llm,
            embed_model=Settings.embed_model,
        )
        summary_query_engine = summary_index.as_query_engine()
        summary_tool = QueryEngineTool(
            query_engine=summary_query_engine,
            metadata=ToolMetadata(
                name="summary_tool",
                description=(
                    "Use this tool for any user questions that ask "
                    "for a summarization of content"
                ),
            ),
        )
        all_tools.append(summary_tool)

    # then we add tools
    all_tools.extend(additional_tools)

    # build agent
    if system_prompt is None:
        return "System prompt not set yet. Please set system prompt first."

    agent = load_agent(
        all_tools,
        llm=Settings.llm,
        system_prompt=system_prompt,
        verbose=True,
        extra_kwargs={"vector_index": vector_index, "rag_params": rag_params},
    )
    return agent, extra_info


def get_web_agent_tool() -> QueryEngineTool:
    """Get web agent tool.

    Wrap with our load and search tool spec.

    """
    from llama_hub.tools.metaphor.base import MetaphorToolSpec

    # TODO: set metaphor API key
    metaphor_tool = MetaphorToolSpec(
        api_key=st.secrets.metaphor_key,
    )
    metaphor_tool_list = metaphor_tool.to_tool_list()

    # TODO: LoadAndSearch doesn't work yet
    # The search_and_retrieve_documents tool is the third in the tool list,
    # as seen above
    # wrapped_retrieve = LoadAndSearchToolSpec.from_defaults(
    #     metaphor_tool_list[2],
    # )

    # NOTE: requires openai right now
    # We don't give the Agent our unwrapped retrieve document tools
    # instead passing the wrapped tools
    web_agent = OpenAIAgent.from_tools(
        # [*wrapped_retrieve.to_tool_list(), metaphor_tool_list[4]],
        metaphor_tool_list,
        llm=BUILDER_LLM,
        verbose=True,
    )

    # return agent as a tool
    # TODO: tune description
    web_agent_tool = QueryEngineTool.from_defaults(
        web_agent,
        name="web_agent",
        description="""
            This agent can answer questions by searching the web. \
Use this tool if the answer is ONLY likely to be found by searching \
the internet, especially for queries about recent events.
        """,
    )

    return web_agent_tool


def get_tool_objects(tool_names: List[str]) -> List:
    """Get tool objects from tool names."""
    # construct additional tools
    tool_objs = []
    for tool_name in tool_names:
        if tool_name == "web_search":
            # build web agent
            tool_objs.append(get_web_agent_tool())
        else:
            raise ValueError(f"Tool {tool_name} not recognized.")

    return tool_objs


class MultimodalChatEngine(BaseChatEngine):
    """Multimodal chat engine.

    This chat engine is a light wrapper around a query engine.
    Offers no real 'chat' functionality, is a beta feature.

    """

    def __init__(self, mm_query_engine: SimpleMultiModalQueryEngine) -> None:
        """Init params."""
        self._mm_query_engine = mm_query_engine

    def reset(self) -> None:
        """Reset conversation state."""
        pass

    @property
    def chat_history(self) -> List[ChatMessage]:
        return []

    @trace_method("chat")
    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        """Main chat interface."""
        # just return the top-k results
        response = self._mm_query_engine.query(message)
        return AgentChatResponse(
            response=str(response), source_nodes=response.source_nodes
        )

    @trace_method("chat")
    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        """Stream chat interface."""
        response = self._mm_query_engine.query(message)

        def _chat_stream(response: str) -> Generator[ChatResponse, None, None]:
            yield ChatResponse(message=ChatMessage(role="assistant", content=response))

        chat_stream = _chat_stream(str(response))
        return StreamingAgentChatResponse(
            chat_stream=chat_stream, source_nodes=response.source_nodes
        )

    @trace_method("chat")
    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        """Async version of main chat interface."""
        response = await self._mm_query_engine.aquery(message)
        return AgentChatResponse(
            response=str(response), source_nodes=response.source_nodes
        )

    @trace_method("chat")
    async def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        """Async version of main chat interface."""
        return self.stream_chat(message, chat_history)


def construct_agent(
    system_prompt: str,
    rag_params: RAGParams,
    docs: List[Document],
    vector_index: Optional[VectorStoreIndex] = None,
    additional_tools: Optional[List] = None,
) -> Tuple[BaseChatEngine, Dict]:
    """Construct agent from docs / parameters / indices."""
    extra_info = {}
    additional_tools = additional_tools or []

    # Resolve embed model and LLM
    embed_model = resolve_embed_model(rag_params.embed_model)
    llm = _resolve_llm(rag_params.llm)

    # Initialize Settings
    from llama_index.core.settings import Settings
    # settings = Settings(
    #     chunk_size=rag_params.chunk_size,
    #     llm=llm,
    #     embed_model=embed_model,
    # )
    print(f"embed_model1: {embed_model}")  # Debug
    print(f"Settings.embed_model1: {Settings.embed_model}")  # Debug
    print(f"rag_params.embed_model1: {rag_params.embed_model}")  # Debug
    print(f"rag_params.llm1: {rag_params.llm}")  # Debug
    print(f"Settings.llm1: {Settings.llm}")  # Debug

    # Create or use existing vector index
    if vector_index is None:
        vector_index = VectorStoreIndex.from_documents(
            docs,
            chunk_size=Settings.chunk_size,
            llm=Settings.llm,
            embed_model=Settings.embed_model,
        )

    extra_info["vector_index"] = vector_index

    # Create vector query engine
    vector_query_engine = vector_index.as_query_engine(
        similarity_top_k=rag_params.top_k
    )

    # Create tools
    all_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="vector_tool",
                description="Use this tool to answer any user question over any data.",
            ),
        )
    ]

    # Add summarization tool if needed
    if rag_params.include_summarization:
        summary_index = SummaryIndex.from_documents(
            docs,
            chunk_size=Settings.chunk_size,
            llm=Settings.llm,
            embed_model=Settings.embed_model,
        )
        summary_query_engine = summary_index.as_query_engine()
        summary_tool = QueryEngineTool(
            query_engine=summary_query_engine,
            metadata=ToolMetadata(
                name="summary_tool",
                description="Use this tool for summarization of content.",
            ),
        )
        all_tools.append(summary_tool)

    # Add additional tools
    all_tools.extend(additional_tools)

    # Create agent
    if not system_prompt:
        raise ValueError("System prompt not set yet. Please set system prompt first.")

    agent = load_agent(
        tools=all_tools,
        llm=Settings.llm,  # Sử dụng `llm` từ settings.
        system_prompt=system_prompt,
        verbose=True,
        extra_kwargs={"vector_index": vector_index, "rag_params": rag_params},
    )

    return agent, extra_info


def get_image_and_text_nodes(
    nodes: List[NodeWithScore],
) -> Tuple[List[NodeWithScore], List[NodeWithScore]]:
    image_nodes = []
    text_nodes = []
    for res_node in nodes:
        if isinstance(res_node.node, ImageNode):
            image_nodes.append(res_node)
        else:
            text_nodes.append(res_node)
    return image_nodes, text_nodes
