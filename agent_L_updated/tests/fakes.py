import re


class FakeDoc:
    def __init__(self, page_content, metadata):
        self.page_content, self.metadata = page_content, metadata


class FakeStore:
    """Mimics the langchain-chroma calls the code uses. 'Dense' ranking is crude word overlap, which is enough to test plumbing."""

    def __init__(self):
        self.ids, self.docs = [], []

    def add_documents(self, documents, ids):
        for i, d in zip(ids, documents):
            self.ids.append(i)
            self.docs.append(d)

    def _match(self, meta, where):
        if not where:
            return True
        for key, cond in where.items():
            if isinstance(cond, dict):
                if meta.get(key) not in cond["$in"]:
                    return False
            elif meta.get(key) != cond:
                return False
        return True

    def get(self, ids=None, where=None, include=None, limit=None):
        pick = [(i, d) for i, d in zip(self.ids, self.docs) if (ids is None or i in ids) and self._match(d.metadata, where)]
        return {"ids": [i for i, _ in pick], "documents": [d.page_content for _, d in pick], "metadatas": [d.metadata for _, d in pick]}

    def similarity_search(self, query, k=4, filter=None):
        q = set(re.findall(r"\w+", query.lower()))
        scored = []
        for d in self.docs:
            if self._match(d.metadata, filter):
                words = set(re.findall(r"\w+", d.page_content.lower()))
                scored.append((len(q & words) / (len(q) or 1), d))
        scored.sort(key=lambda x: -x[0])
        return [d for _, d in scored[:k]]


def _delete(self, ids):
    keep = [(i, d) for i, d in zip(self.ids, self.docs) if i not in set(ids)]
    self.ids[:] = [i for i, _ in keep]
    self.docs[:] = [d for _, d in keep]


FakeStore.delete = _delete
