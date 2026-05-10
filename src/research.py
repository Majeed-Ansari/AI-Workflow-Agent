import arxiv


def search_arxiv_papers(query, max_results=5):

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []

    for result in search.results():

        papers.append(
            {
                "title": result.title,
                "summary": result.summary,
                "authors": [
                    author.name
                    for author in result.authors
                ],
                "pdf_url": result.pdf_url
            }
        )

    return papers