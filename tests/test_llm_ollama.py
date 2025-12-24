import json
from unittest.mock import Mock, patch, MagicMock
from urllib.error import URLError, HTTPError
from terrance_teacher.llm.ollama import OllamaClient


def test_generate_success():
    """Test successful Ollama generation."""
    mock_response_data = {
        "model": "llama3.2",
        "response": "This is the generated text",
        "done": True,
    }
    
    mock_response = Mock()
    mock_response.read.return_value = json.dumps(mock_response_data).encode("utf-8")
    mock_response.__enter__ = Mock(return_value=mock_response)
    mock_response.__exit__ = Mock(return_value=False)
    
    with patch("urllib.request.urlopen", return_value=mock_response):
        client = OllamaClient()
        result = client.generate("test prompt")
    
    assert result == "This is the generated text"


def test_generate_connection_error():
    """Test handling of connection errors."""
    with patch("urllib.request.urlopen", side_effect=URLError("Connection refused")):
        client = OllamaClient()
        result = client.generate("test prompt")
    
    assert result is None


def test_generate_http_error():
    """Test handling of HTTP errors."""
    with patch("urllib.request.urlopen", side_effect=HTTPError("url", 500, "Internal Server Error", {}, None)):
        client = OllamaClient()
        result = client.generate("test prompt")
    
    assert result is None


def test_generate_timeout():
    """Test handling of timeout errors."""
    with patch("urllib.request.urlopen", side_effect=TimeoutError("Request timed out")):
        client = OllamaClient()
        result = client.generate("test prompt")
    
    assert result is None


def test_generate_invalid_json():
    """Test handling of invalid JSON response."""
    mock_response = Mock()
    mock_response.read.return_value = b"not valid json"
    mock_response.__enter__ = Mock(return_value=mock_response)
    mock_response.__exit__ = Mock(return_value=False)
    
    with patch("urllib.request.urlopen", return_value=mock_response):
        client = OllamaClient()
        result = client.generate("test prompt")
    
    assert result is None


def test_build_request_payload():
    """Test that request payload is built correctly."""
    mock_response_data = {
        "model": "llama3.2",
        "response": "test",
        "done": True,
    }
    
    mock_response = Mock()
    mock_response.read.return_value = json.dumps(mock_response_data).encode("utf-8")
    mock_response.__enter__ = Mock(return_value=mock_response)
    mock_response.__exit__ = Mock(return_value=False)
    
    captured_data = None
    
    def capture_request(*args, **kwargs):
        if args and len(args) > 1:
            captured_data = args[1]
        elif 'data' in kwargs:
            captured_data = kwargs['data']
        return mock_response
    
    with patch("urllib.request.urlopen", side_effect=capture_request) as mock_urlopen:
        client = OllamaClient(base_url="http://localhost:11434", model="llama3.2")
        client.generate("test prompt")
        
        # Verify request was made
        assert mock_urlopen.called
        call_args = mock_urlopen.call_args
        
        # Get request object
        if call_args[0]:
            request = call_args[0][0]
            assert request.get_full_url() == "http://localhost:11434/api/generate"
            # Check headers
            headers = request.headers
            content_type = headers.get("Content-type") or headers.get("Content-Type")
            assert content_type == "application/json"
            
            # Get data from request
            if hasattr(request, 'data'):
                data = json.loads(request.data.decode("utf-8"))
            else:
                # Try to get from call args
                if len(call_args[0]) > 1:
                    data = json.loads(call_args[0][1].decode("utf-8"))
                else:
                    data = json.loads(call_args[1].get('data', b'{}').decode("utf-8"))
        else:
            # Keyword args only
            data = json.loads(call_args[1]['data'].decode("utf-8"))
        
        assert data["model"] == "llama3.2"
        assert data["prompt"] == "test prompt"
        assert data["stream"] is False

