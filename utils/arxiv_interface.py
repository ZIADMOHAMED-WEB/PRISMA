import arxiv
from datetime import datetime

def search_arxiv(query, from_year, to_year, max_results=10):
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending
    )

    results = []
    for paper in client.results(search):
        pub_year = paper.published.year
        if from_year <= pub_year <= to_year:
            results.append(paper)
        if len(results) >= max_results:
            break

    return results