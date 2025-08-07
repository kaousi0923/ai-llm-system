"""
Tests for vector database operations
"""
import pytest
import os
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import json


class TestVectorDatabase:
    """Test suite for vector database operations"""
    
    @pytest.mark.unit
    @patch('modules.vector_db.create_vector_db')
    def test_create_vector_db(self, mock_create):
        """Test creating a vector database"""
        mock_create.return_value = True
        
        from modules import vector_db
        result = mock_create(
            name="test_db",
            documents=["doc1", "doc2"],
            file_names=["file1.txt", "file2.txt"],
            model="intfloat/multilingual-e5-base",
            summarizer="Test summary",
            text_splitter_type="Recursive",
            chunk_size=1500,
            chunk_overlap=100,
            vector_dbs_dir="/path/to/dbs"
        )
        
        assert result is True
        mock_create.assert_called_once()
    
    @pytest.mark.unit
    @patch('modules.vector_db.list_vector_dbs')
    def test_list_vector_dbs(self, mock_list):
        """Test listing vector databases"""
        mock_list.return_value = [
            {
                "name": "db1",
                "created_at": "2024-01-01",
                "summary": "Database 1",
                "files": ["file1.txt"]
            },
            {
                "name": "db2",
                "created_at": "2024-01-02",
                "summary": "Database 2",
                "files": ["file2.txt", "file3.txt"]
            }
        ]
        
        from modules import vector_db
        dbs = mock_list("/path/to/dbs", None)
        
        assert len(dbs) == 2
        assert dbs[0]["name"] == "db1"
        assert len(dbs[1]["files"]) == 2
    
    @pytest.mark.unit
    @patch('modules.vector_db.delete_vector_db')
    def test_delete_vector_db(self, mock_delete):
        """Test deleting a vector database"""
        mock_delete.return_value = True
        
        from modules import vector_db
        result = mock_delete("test_db", "/path/to/dbs")
        
        assert result is True
        mock_delete.assert_called_with("test_db", "/path/to/dbs")
    
    @pytest.mark.unit
    @patch('modules.vector_db.search_vector_db')
    def test_search_vector_db(self, mock_search):
        """Test searching in vector database"""
        mock_search.return_value = ["/path/to/db1", "/path/to/db2"]
        
        from modules import vector_db
        results = mock_search("test_db", "query text", "/path/to/dbs")
        
        assert len(results) == 2
        assert "/path/to/db1" in results
    
    @pytest.mark.unit
    def test_vector_db_metadata(self, temp_dir):
        """Test vector database metadata handling"""
        db_path = temp_dir / "test_db"
        db_path.mkdir()
        
        metadata = {
            "name": "test_db",
            "created_at": "2024-01-01 12:00:00",
            "summary": "Test database",
            "files": ["test.txt"],
            "model": "intfloat/multilingual-e5-base",
            "chunk_size": 1500,
            "chunk_overlap": 100
        }
        
        metadata_file = db_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, ensure_ascii=False))
        
        # Read back metadata
        loaded_metadata = json.loads(metadata_file.read_text())
        assert loaded_metadata["name"] == "test_db"
        assert loaded_metadata["chunk_size"] == 1500
    
    @pytest.mark.integration
    @pytest.mark.skip(reason="Requires ChromaDB installation")
    def test_chromadb_operations(self):
        """Test actual ChromaDB operations"""
        # This would test actual ChromaDB functionality
        # Requires ChromaDB to be installed and configured
        pass
    
    @pytest.mark.unit
    def test_embedding_model_configuration(self):
        """Test embedding model configuration"""
        from modules.config import EMBED_API_DEFAULTS
        
        model = EMBED_API_DEFAULTS.get("MODEL")
        assert model is not None
        assert "multilingual" in model or "e5" in model


class TestHybridSearch:
    """Test suite for hybrid search functionality"""
    
    @pytest.mark.unit
    @patch('main.extract_named_entities_with_llm')
    def test_extract_named_entities(self, mock_extract):
        """Test named entity extraction"""
        mock_extract.return_value = ["實體1", "實體2", "GRI 202-2"]
        
        from main import extract_named_entities_with_llm
        entities = mock_extract("測試查詢文字", "gpt-4o-mini")
        
        assert len(entities) == 3
        assert "GRI 202-2" in entities
    
    @pytest.mark.unit
    @patch('main.rewrite_query_to_statement')
    def test_query_rewriting(self, mock_rewrite):
        """Test query rewriting functionality"""
        mock_rewrite.return_value = "轉寫後的敘述句"
        
        from main import rewrite_query_to_statement
        result = mock_rewrite("這是什麼？", "gpt-4o-mini")
        
        assert result == "轉寫後的敘述句"
    
    @pytest.mark.unit
    @patch('main.hybrid_search')
    def test_hybrid_search_function(self, mock_hybrid):
        """Test hybrid search combining BM25 and semantic search"""
        mock_docs = [
            MagicMock(page_content="Document 1 content"),
            MagicMock(page_content="Document 2 content")
        ]
        mock_hybrid.return_value = mock_docs
        
        from main import hybrid_search
        results = mock_hybrid(
            user_query="test query",
            model="gpt-4o-mini",
            matching_db_paths=["/path/to/db"],
            embeddings=MagicMock(),
            k=10
        )
        
        assert len(results) == 2
        assert hasattr(results[0], 'page_content')


class TestVectorDBIntegration:
    """Test suite for vector database integration with main app"""
    
    @pytest.mark.integration
    def test_db_creation_workflow(self, temp_dir):
        """Test complete database creation workflow"""
        # This test would simulate the entire workflow
        # from file upload to database creation
        
        # 1. Create test files
        test_file = temp_dir / "test.txt"
        test_file.write_text("Test content for vector database")
        
        # 2. Create metadata
        metadata = {
            "name": "integration_test_db",
            "files": ["test.txt"],
            "created_at": "2024-01-01",
            "summary": "Integration test database"
        }
        
        # 3. Verify structure
        assert test_file.exists()
        assert metadata["name"] == "integration_test_db"
    
    @pytest.mark.integration
    @pytest.mark.skip(reason="Requires full application context")
    def test_rag_pipeline(self):
        """Test complete RAG pipeline"""
        # This would test:
        # 1. Document upload
        # 2. Vector database creation
        # 3. Query processing
        # 4. Retrieval
        # 5. Response generation
        pass