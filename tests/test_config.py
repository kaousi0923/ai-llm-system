"""
Tests for configuration module
"""
import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open
import configparser


class TestConfigModule:
    """Test suite for configuration management"""
    
    @pytest.mark.unit
    def test_config_file_loading(self):
        """Test that configuration file can be loaded"""
        from modules import config
        
        assert hasattr(config, 'DEFAULT')
        assert hasattr(config, 'EMBED_API_DEFAULTS')
        assert hasattr(config, 'CHAT_API_DEFAULTS')
    
    @pytest.mark.unit
    def test_default_config_values(self):
        """Test default configuration values"""
        from modules import config
        
        # Check some essential config values exist
        assert config.DEFAULT.get('LOGS_DIR') is not None
        assert config.DEFAULT.get('VECTOR_DBS_DIR') is not None
        assert config.DEFAULT.get('TEMP_DIR') is not None
    
    @pytest.mark.unit
    def test_embed_api_defaults(self):
        """Test embedding API default configuration"""
        from modules import config
        
        assert config.EMBED_API_DEFAULTS.get('MODEL') is not None
        assert config.EMBED_API_DEFAULTS.get('CHUNK_SIZE') is not None
        assert config.EMBED_API_DEFAULTS.get('CHUNK_OVERLAP') is not None
        
        # Test that numeric values can be converted
        chunk_size = int(config.EMBED_API_DEFAULTS.get('CHUNK_SIZE'))
        assert chunk_size > 0
    
    @pytest.mark.unit
    def test_chat_api_defaults(self):
        """Test chat API default configuration"""
        from modules import config
        
        assert config.CHAT_API_DEFAULTS.get('MODEL') is not None
        assert config.CHAT_API_DEFAULTS.get('TEMPERATURE') is not None
        assert config.CHAT_API_DEFAULTS.get('MAX_TOKENS') is not None
        
        # Test that numeric values are in valid ranges
        temperature = float(config.CHAT_API_DEFAULTS.get('TEMPERATURE'))
        assert 0 <= temperature <= 2
        
        max_tokens = int(config.CHAT_API_DEFAULTS.get('MAX_TOKENS'))
        assert max_tokens > 0
    
    @pytest.mark.unit
    def test_config_file_path(self):
        """Test that config file path is correctly constructed"""
        from modules import config
        
        assert config.CONFIG_FILE.endswith('config.ini')
        assert os.path.dirname(config.CONFIG_FILE) == config.PROJECT_DIR
    
    @pytest.mark.unit
    def test_missing_config_file_handling(self):
        """Test behavior when config file is missing"""
        with patch('configparser.ConfigParser.read') as mock_read:
            mock_read.return_value = []
            
            # Re-import to trigger the read
            import importlib
            import modules.config
            importlib.reload(modules.config)
            
            # Should still have the sections (even if empty)
            assert hasattr(modules.config, 'DEFAULT')
    
    @pytest.mark.unit
    def test_config_encoding(self):
        """Test that config file is read with correct encoding"""
        sample_config = """[DEFAULT]
test_key = 測試值
unicode_test = 你好世界"""
        
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', 
                                        suffix='.ini', delete=False) as f:
            f.write(sample_config)
            temp_path = f.name
        
        try:
            config = configparser.ConfigParser()
            config.read(temp_path, encoding='utf-8')
            
            assert config['DEFAULT']['test_key'] == '測試值'
            assert config['DEFAULT']['unicode_test'] == '你好世界'
        finally:
            os.unlink(temp_path)


class TestConfigUtilities:
    """Test configuration utility functions from main.py"""
    
    @pytest.mark.unit
    def test_parse_optional_int(self):
        """Test parse_optional_int function"""
        from main import parse_optional_int
        
        assert parse_optional_int("123", 0) == 123
        assert parse_optional_int("", 100) == 100
        assert parse_optional_int(None, 50) == 50
        assert parse_optional_int("invalid", 25) == 25
        assert parse_optional_int("  ", 10) == 10
    
    @pytest.mark.unit
    def test_parse_optional_float(self):
        """Test parse_optional_float function"""
        from main import parse_optional_float
        
        assert parse_optional_float("1.23", 0.0) == 1.23
        assert parse_optional_float("", 0.5) == 0.5
        assert parse_optional_float(None, 0.1) == 0.1
        assert parse_optional_float("invalid", 0.2) == 0.2
        assert parse_optional_float("  ", 0.3) == 0.3


class TestEnvironmentVariables:
    """Test environment variable configuration"""
    
    @pytest.mark.unit
    def test_required_env_vars(self, monkeypatch):
        """Test that required environment variables are set"""
        # These should be set by the setup_test_env fixture
        assert os.getenv('PYTHONUTF8') == '1'
        assert os.getenv('DISABLE_TORCH_INDUCTOR') == '1'
        assert os.getenv('TORCH_COMPILE') == '0'
    
    @pytest.mark.unit
    def test_tunnel_port_env(self, monkeypatch):
        """Test tunnel port environment variable"""
        monkeypatch.setenv('TUNNEL_PORT', '6000')
        assert os.getenv('TUNNEL_PORT') == '6000'
        
        # Test default value when not set
        monkeypatch.delenv('TUNNEL_PORT', raising=False)
        from tunnel_setup import port
        # Should use default value from the script