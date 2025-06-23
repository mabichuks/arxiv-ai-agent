from llama_index.core import (
    Document,
    Settings,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage
)

from tools import fetch_arxiv_papers

class IndexManager:

    def __init__(self, embed_model):
        self.embed_model = embed_model
        self.papers = []
        
    def fetch_papers(self, title: str, papers_count: int = 10):
        self.papers = fetch_arxiv_papers(title, max_results=papers_count)
    
    def create_document_from_paper(self, paper: dict):
        """
        Create a Document object from a paper dictionary.
        
        Args:
            paper (dict): A dictionary containing paper details.
        
        Returns:
            Document: A Document object containing the paper's content.
        """
        for paper in self.papers:
            content = (
                f"Title: {paper['title']}\n"
                f"Authors: {', '.join(paper['authors'])}\n"
                f"Summary: {paper['summary']}\n"
                f"Published: {paper['published']}\n"
                f"Journal Reference: {paper['journal_ref']}\n"
                f"DOI: {paper['doi']}\n"
                f"Primary Category: {paper['primary_category']}\n"
                f"Categories: {', '.join(paper['categories'])}\n"
                f"PDF URL: {paper['pdf_url']}\n"
                f"Arxiv URL: {paper['arxiv_url']}\n"
            )
            self.documents.append(Document(content=content))

    def create_index(self):
        self.documents = []
        self.create_document_from_paper(self.papers)
        Settings.chunk_size = 1024
        Settings.chunk_overlap = 20

        index = VectorStoreIndex.from_documents(self.documents, 
                                                embed_model=self.embed_model)

    def retrieve_index(self, index_name: str = "index/") -> VectorStoreIndex:
        """
        Retrieve an existing index from storage.
        
        Args:
            index_name (str): The name of the index to retrieve.
        
        Returns:
            VectorStoreIndex: The retrieved index.
        """
        storage_context = StorageContext.from_defaults(persist_dir=index_name)
        return load_index_from_storage(storage_context)
    
    def list_papers(self):
        """
        List the titles of the fetched papers.
        
        Returns:
            list: A list of paper titles.
        """
        print([paper['title'] for paper in self.papers]) if self.papers else print("No papers fetched yet.")