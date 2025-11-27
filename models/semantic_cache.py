import json
import time
from typing import Dict, List, Optional, Tuple
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from models.base_cache import CacheResult
from models.llm_model import LLMModel
from models.storage_model import StorageModel
from config.config import config
from utils.utils import LogUtils, cosine_sim
from timebudget import timebudget

logger = LogUtils().get_logging()


class SemanticCache:
    """
    SemanticCache provides an interface for semantic caching with vector-based lookup,
    optional LLM fallback, and data ingestion from DataFrames.

    This class supports efficient semantic search over a cache of question-answer pairs,
    fallback to an LLM for unseen queries, and flexible data ingestion.

    Attributes:
        cache (StorageModel): The storage backend for the cache.
        llm (LLMModel): The language model used for fallback responses.
        similarity_threshold (float): Minimum cosine similarity for semantic cache hits.
    """

    def __init__(
        self,
        name: str = "semantic-cache",
        storage: str = "memory",
        llm: str = "local",
        similarity_threshold: float = 0.3,
        encode_model: str = "all-mpnet-base-v2",
        ttl: int = 3600,
    ):
        """
        Initialize a new SemanticCache.

        Args:
            name (str): Name of the cache.
            storage (str): Storage type ("memory", "redis", etc).
            llm (str): LLM engine type.
            similarity_threshold (float): Min cosine similarity for a cache hit.
            encode_model (str): Name of sentence embedding model.
            ttl (int): Time-to-live (seconds) for cache entries.
        """
        self.cache = StorageModel(name, storage, encode_model, ttl)
        self.llm = LLMModel(llm)
        self.similarity_threshold = similarity_threshold
        logger.info("storage type {0} with ttl {1}".format(storage, ttl))
        logger.info("llm type {0}".format(llm))
        logger.info("similarity threshold {0}".format(similarity_threshold))

    @classmethod
    def from_config(
        cls,
        config,
    ) -> "SemanticCacheWrapper":
        """
        Factory method to construct SemanticCache from a config dictionary.

        Args:
            config (dict): Configuration containing initialization parameters.

        Returns:
            SemanticCacheWrapper: Configured instance of SemanticCache.
        """
        return cls(
            name=config["name"],
            storage=config["storage_type"],
            llm=config["llm_type"],
            similarity_threshold=float(config["similarity_threshold"]),
            encode_model=config["encode_model"],
            ttl=int(config["ttl_seconds"]),
        )

    def set_question_to_id(self, question_to_id: Dict[str, int]):
        """
        Set mapping from questions to unique IDs.

        Args:
            question_to_id (dict): Dictionary mapping questions to integer IDs.
        """
        self.question_to_id = question_to_id

    def ingest_from_df(
        self,
        df: pd.DataFrame,
        *,
        q_col: str = "question",
        a_col: str = "answer",
        clear: bool = True,
        ttl_override: Optional[int] = None,
        return_id_map: bool = False,
    ) -> Optional[Dict[str, int]]:
        """
        Ingest a DataFrame of question/answer pairs into the cache.

        Args:
            df (pd.DataFrame): DataFrame containing questions and answers.
            q_col (str): Column name for question texts. Defaults to "question".
            a_col (str): Column name for answer texts. Defaults to "answer".
            clear (bool): Whether to clear cache before ingest. (Unused)
            ttl_override (Optional[int]): Override TTL for entries. (Unused)
            return_id_map (bool): Whether to return a mapping of questions to row IDs.

        Returns:
            Optional[Dict[str, int]]: Map of questions to IDs if return_id_map is True; else None.
        """
        question_to_id: Dict[str, int] = {}
        idx = 0
        for row in df[[q_col, a_col]].itertuples(index=False, name=None):
            q, a = row
            self.cache.store(q, a)

    def semantic_search(self, query: str, num_results: int = 1) -> tuple:
        """
        Search the cache for questions most semantically similar to the query.

        Args:
            query (str): Query string.
            num_results (int): Number of similar results to return.

        Returns:
            tuple: List of (question, similarity) tuples, sorted by similarity descending.
        """
        query_embedding = self.cache.encoder.encode([query])[0]
        results = []
        for key, value in self.cache.get_all_key_values():
            value = json.loads(value)
            similarity = cosine_sim(query_embedding, value.get("embedding"))
            if similarity >= self.similarity_threshold:
                results.append((key, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        if len(results) >= num_results:
            results = results[:num_results]
        return results

    def check_cache(self, query: str, num_results: int = 1):
        """
        Perform a semantic cache lookup for the specified query.

        Args:
            query (str): Query string to look up.
            num_results (int): Maximum number of results to fetch.

        Returns:
            list: List of dictionaries containing response and metadata.
        """
        results = []
        semantic_search_results = self.semantic_search(query, num_results)
        for semantic_search_result in semantic_search_results:
            question = semantic_search_result[0]
            similarity = semantic_search_result[1]
            start_time = time.perf_counter()
            answer = self.cache.get_value_by_key(question).get("answer")
            latency = time.perf_counter() - start_time

            meta_data = {
                "source": "cache",
                "vector_similarity": float(similarity),
                "prompt": question,
                "latency": latency,
            }
            result_data = {
                "response": answer,
                "meta_data": meta_data,
            }
            results.append(result_data)
        return results

    def get_llm_results(self, query: str, num_results: int = 1):
        """
        Get results from the LLM as a fallback if the cache misses.

        Args:
            query (str): Query string.
            num_results (int): Ignored, always returns one result.

        Returns:
            list: List containing a single result dictionary (response, metadata).
        """
        start_time = time.perf_counter()
        answer = self.llm.get_llm_response(query)
        latency = time.perf_counter() - start_time

        q_emb = self.cache.encoder.encode(query)
        a_emb = self.cache.encoder.encode([answer])[0]
        similarity = cosine_sim(q_emb, a_emb)
        meta_data = {
            "source": "llm",
            "vector_similarity": float(similarity),
            "prompt": query,
            "latency": latency,
        }
        result_data = {
            "response": answer,
            "meta_data": meta_data,
        }
        self.cache.store(query, answer)
        return [result_data]

    def check(
        self,
        query: str,
        forceRefresh: bool = False,
        num_results: int = 1,
    ) -> List[CacheResult]:
        """
        Check semantic cache and return results for a single query.
        Optionally forces an LLM refresh.

        Args:
            query (str): Query string to search for.
            forceRefresh (bool): If True, skip cache and go straight to LLM.
            num_results (int): Maximum number of results to return.

        Returns:
            List[CacheResult]: List of semantic cache or LLM results.
        """
        number_candidates = 1
        candidates = []
        if not forceRefresh:
            logger.info("forceRefresh false")
            candidates = self.check_cache(query, number_candidates)
        else:
            candidates = self.get_llm_results(query, num_results)

        if len(candidates) < number_candidates:
            candidates = self.get_llm_results(query, num_results)

        result = {}
        if len(candidates) >= num_results:
            result = candidates[:num_results]

        return result
