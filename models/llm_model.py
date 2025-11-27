from typing import Optional, Any
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from config.config import config, load_openai_key
from utils.utils import LogUtils

logger = LogUtils().get_logging()


class LLMModel:
    """
    LLMModel provides a unified interface to interact with language models using the LangChain library.
    The model can be either a local (Ollama) LLM or an OpenAI-hosted model.

    Attributes:
        llm (Any): The instantiated language model client (ChatOpenAI or ChatOllama).
    """

    def __init__(self, type: str = "local") -> None:
        """
        Initialize the LLMModel.

        Args:
            type (str, optional): Specifies which LLM backend to use ('openai' for OpenAI or
            'local' for local Ollama). Defaults to "local".
        """
        self.llm: Optional[Any] = None
        if type == "openai":
            load_openai_key()
            MODEL_NAME = "llama3.2:1b"
            self.llm = ChatOpenAI(
                model=MODEL_NAME,
                temperature=0.1,
                max_tokens=150,
            )
        else:
            MODEL_NAME = "llama3.2:1b"
            self.llm = ChatOllama(
                model=MODEL_NAME,
                temperature=0.1,
                max_tokens=150,
            )
        logger.info("{} LLM set".format(type))

    def get_llm_response(self, question: str) -> str:
        """
        Generate a response from the language model given a code documentation prompt.

        Args:
            question (str): The code or query for which documentation is required.

        Returns:
            str: The language model's generated documentation as a string.
        """
        response = self.llm.invoke([HumanMessage(content=question)])
        return response.content.strip()
