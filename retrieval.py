"""
retrieval.py

Loads the scraped catalog and builds a TF-IDF index over it.
On each chat turn, takes the conversation history and returns
the most relevant assessments as candidates for the agent.

I went with TF-IDF over sentence-transformers because it's
fast, needs no GPU, and fits inside the 30-second response
budget easily. The trade-off is weaker semantic matching — 
explained in the write-up.
"""

import json
import logging
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "catalog.json")


class CatalogRetriever:
    def __init__(self, catalog_path: str = CATALOG_PATH):
        with open(catalog_path, encoding="utf-8") as f:
            self.catalog = json.load(f)

        # build one text blob per item: name + test_type + description
        # weighting name more by repeating it helps recall a lot
        docs = []
        for item in self.catalog:
            name = item.get("name", "")
            test_type = item.get("test_type", "")
            desc = item.get("description", "")
            docs.append(f"{name} {name} {test_type} {desc}")

        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(docs)

        logger.info("Catalog loaded: %d items", len(self.catalog))

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Return up to top_k catalog items most relevant to query."""
        if not query.strip():
            return []

        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        # grab top results, skip anything with zero overlap
        ranked = np.argsort(scores)[::-1]
        results = []
        for idx in ranked[:top_k]:
            if scores[idx] > 0.0:
                results.append(self.catalog[idx])

        return results

    def get_by_url(self, url: str) -> dict | None:
        """Look up a single catalog item by URL. Used for URL validation."""
        url = url.rstrip("/") + "/"
        for item in self.catalog:
            if item["url"].rstrip("/") + "/" == url:
                return item
        return None

    def all_urls(self) -> set[str]:
        """All valid URLs in the catalog. Agent uses this to strip hallucinated URLs."""
        return {item["url"].rstrip("/") + "/" for item in self.catalog}

    def build_query(self, messages: list[dict]) -> str:
        """
        Builds a retrieval query from conversation history.

        Just concatenates user messages — no fancy summarization.
        Recent turns get a bit more weight by appearing at the end,
        which TF-IDF treats as slightly higher frequency.
        """
        user_turns = [m["content"] for m in messages if m["role"] == "user"]
        if not user_turns:
            return ""
        # give more weight to the last 2 user messages
        recent = user_turns[-2:]
        earlier = user_turns[:-2]
        return " ".join(earlier) + " " + " ".join(recent) + " " + " ".join(recent)
