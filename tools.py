import arxiv
import requests
import os

client = arxiv.Client()

def fetch_arxiv_papers(title: str, max_results: int):
    """
    Fetches a list of arXiv papers based on the given title.

    Args:
        title (str): The title or keywords to search for.
        max_results (int): The maximum number of results to return.

    Returns:
        list: A list of dictionaries containing paper details.
    """
    search = arxiv.Search(
        query=title,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    papers = []
    results = client.results(search)

    for result in results:
        paper_info ={
            'title': result.title,
            'summary': result.summary,
            'published': result.published.isoformat(),
            'authors': [author.name for author in result.authors],
            'journal_ref': result.journal_ref,
            'doi': result.doi,
            'primary_category': result.primary_category,
            'categories': result.categories,
            'pdf_url': result.pdf_url,
            'arxiv_url': result.entry_id
        }
        papers.append(paper_info)
    return papers

def download_pdf(pdf_url: str, output_file_name: str):
    """
    Downloads a PDF file from the given URL.

    Args:
        pdf_url (str): The URL of the PDF file.
        output_file_name (str): The name of the file to save the PDF as.

    Returns:
        str: The path to the downloaded PDF file.
    """
    file_name = "papers"
    try:
        os.makedirs(file_name, exist_ok=True)
        output_path = os.path.join(file_name, output_file_name)
        response = requests.get(pdf_url)
        response.raise_for_status()  # Raise an error for bad responses
        with open(output_path, 'wb') as file:
            file.write(response.content)
        return f"PDF downloaded successfully and saved as: {output_path}"
    except requests.exceptions.RequestException as e:
        return f"An error occurred while downloading the PDF: {e}"