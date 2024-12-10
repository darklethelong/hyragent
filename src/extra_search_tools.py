import json
from pathlib import Path
from transformers.agents import tool
from logging_function import AppLogger

logger = AppLogger(__name__, log_file= 'reranker.log')

def load_ka_dictionary(path_dictionary: Path = 'portico_ka_dictionary.json'):
    """Load extra document

    Args:
        path_dictionary (Path, optional): Defaults to 'portico_ka_dictionary.json'.

    """
    try:
        logger.info("Loading extra dictionary")
        return json.load(open(path_dictionary))
    except:
        logger.info("Fail to load extra dictionary")
        return  {}
        
@tool
def extra_search(document_ids: str) -> str:
    """Search cited documents

    Args:
        document_ids: ids document. Example: KB0129575, KB0207589 

    Returns:
        The content of documents
    """
    try:
        dict_document = load_ka_dictionary()
        list_ids_document = document_ids.split(",")
        content = ""
        for id in list_ids_document:
            try:
                content += dict_document[id] +"\n\n"
            except:
                logger.info(f"Document {id} deleted")
                continue
        return content

    except:
        return "Not found documents"