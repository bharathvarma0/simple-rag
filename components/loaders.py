"""
Document loaders for various file formats
"""

from pathlib import Path
from typing import List, Any
import sys
sys.path.append(str(Path(__file__).parent.parent))

from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    CSVLoader,
    Docx2txtLoader
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader
from utils.logger import get_logger

logger = get_logger(__name__)


class DocumentLoader:
    """Load documents from various file formats"""
    
    def __init__(self, data_dir: str):
        """
        Initialize document loader
        
        Args:
            data_dir: Directory path containing documents
        """
        self.data_dir = Path(data_dir).resolve()
        logger.info(f"Data directory: {self.data_dir}")
    
    def load_all(self) -> List[Any]:
        """
        Load all supported files from the data directory
        
        Supported formats: PDF, TXT, CSV, Excel (.xlsx), Word (.docx), JSON
        
        Returns:
            List of LangChain Document objects
        """
        documents = []
        
        # PDF files
        pdf_files = list(self.data_dir.glob('**/*.pdf'))
        logger.info(f"Found {len(pdf_files)} PDF files")
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_file))
                loaded = loader.load()
                logger.info(f"Loaded {len(loaded)} pages from {pdf_file.name}")
                documents.extend(loaded)
            except Exception as e:
                logger.error(f"Failed to load PDF {pdf_file.name}: {e}")
        
        # TXT files
        txt_files = list(self.data_dir.glob('**/*.txt'))
        logger.info(f" Found {len(txt_files)} TXT files")
        for txt_file in txt_files:
            try:
                loader = TextLoader(str(txt_file), encoding='utf-8')
                loaded = loader.load()
                logger.info(f" Loaded {len(loaded)} documents from {txt_file.name}")
                documents.extend(loaded)
            except Exception as e:
                logger.error(f" Failed to load TXT {txt_file.name}: {e}")
        
        # CSV files
        csv_files = list(self.data_dir.glob('**/*.csv'))
        logger.info(f" Found {len(csv_files)} CSV files")
        for csv_file in csv_files:
            try:
                loader = CSVLoader(str(csv_file))
                loaded = loader.load()
                logger.info(f" Loaded {len(loaded)} rows from {csv_file.name}")
                documents.extend(loaded)
            except Exception as e:
                logger.error(f" Failed to load CSV {csv_file.name}: {e}")
        
        # Excel files
        xlsx_files = list(self.data_dir.glob('**/*.xlsx'))
        logger.info(f" Found {len(xlsx_files)} Excel files")
        for xlsx_file in xlsx_files:
            try:
                loader = UnstructuredExcelLoader(str(xlsx_file))
                loaded = loader.load()
                logger.info(f" Loaded {len(loaded)} sheets from {xlsx_file.name}")
                documents.extend(loaded)
            except Exception as e:
                logger.error(f" Failed to load Excel {xlsx_file.name}: {e}")
        
        # Word files
        docx_files = list(self.data_dir.glob('**/*.docx'))
        logger.info(f" Found {len(docx_files)} Word files")
        for docx_file in docx_files:
            try:
                loader = Docx2txtLoader(str(docx_file))
                loaded = loader.load()
                logger.info(f" Loaded {len(loaded)} documents from {docx_file.name}")
                documents.extend(loaded)
            except Exception as e:
                logger.error(f" Failed to load Word {docx_file.name}: {e}")
        
        # JSON files
        json_files = list(self.data_dir.glob('**/*.json'))
        logger.info(f" Found {len(json_files)} JSON files")
        for json_file in json_files:
            try:
                loader = JSONLoader(str(json_file))
                loaded = loader.load()
                logger.info(f" Loaded {len(loaded)} documents from {json_file.name}")
                documents.extend(loaded)
            except Exception as e:
                logger.error(f" Failed to load JSON {json_file.name}: {e}")
        
        logger.info(f" Total documents loaded: {len(documents)}")
        return documents

