import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock Blender C modules before importing package
for mod in ["bpy", "bpy.types", "bpy.props", "bpy.app", "bpy.app.timers", "bmesh", "mathutils"]:
    sys.modules[mod] = MagicMock()

# Add current workspace directory to sys.path
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gptblend.preferences import (
    DEFAULT_API_URL,
    get_effective_endpoint,
    get_effective_model,
)
from gptblend.core.client import send_message, GPTBlendError, get_chat_tools


class DummyPrefs:
    def __init__(self, api_endpoint="", model="gpt-6.1-sol", custom_model=""):
        self.api_endpoint = api_endpoint
        self.model = model
        self.custom_model = custom_model


class TestCustomEndpoint(unittest.TestCase):

    def test_get_effective_endpoint_default(self):
        prefs = DummyPrefs(api_endpoint="")
        self.assertEqual(get_effective_endpoint(prefs), DEFAULT_API_URL)

        prefs = DummyPrefs(api_endpoint="   ")
        self.assertEqual(get_effective_endpoint(prefs), DEFAULT_API_URL)

    def test_get_effective_endpoint_normalizes_v1_custom(self):
        prefs = DummyPrefs(api_endpoint="http://localhost:8000/v1")
        self.assertEqual(
            get_effective_endpoint(prefs), "http://localhost:8000/v1/chat/completions"
        )

        prefs = DummyPrefs(api_endpoint="https://my-proxy.com/v1/")
        self.assertEqual(
            get_effective_endpoint(prefs), "https://my-proxy.com/v1/chat/completions"
        )

    def test_get_effective_endpoint_normalizes_v1_openai(self):
        prefs = DummyPrefs(api_endpoint="https://api.openai.com/v1")
        self.assertEqual(
            get_effective_endpoint(prefs), "https://api.openai.com/v1/responses"
        )

    def test_get_effective_model_preset(self):
        prefs = DummyPrefs(model="gpt-6.1-sol")
        self.assertEqual(get_effective_model(prefs), "gpt-6.1-sol")

    def test_get_effective_model_custom(self):
        prefs = DummyPrefs(model="custom", custom_model="gpt-6.1-sol")
        self.assertEqual(get_effective_model(prefs), "gpt-6.1-sol")

        prefs_empty = DummyPrefs(model="custom", custom_model="  ")
        self.assertEqual(get_effective_model(prefs_empty), "gpt-6.1-sol")

    def test_get_chat_tools_format(self):
        chat_tools = get_chat_tools()
        self.assertGreater(len(chat_tools), 0)
        first = chat_tools[0]
        self.assertEqual(first["type"], "function")
        self.assertIn("function", first)
        self.assertIn("name", first["function"])
        self.assertIn("description", first["function"])

    def test_send_message_requires_key_for_openai(self):
        with self.assertRaises(GPTBlendError) as ctx:
            send_message(
                api_key="",
                model="gpt-6.1-sol",
                user_message="Hello",
                endpoint=DEFAULT_API_URL,
            )
        self.assertIn("No OpenAI API key", str(ctx.exception))

    @patch("gptblend.core.client.urllib.request.urlopen")
    def test_send_message_chat_completions(self, mock_urlopen):
        # Mock chat completions response
        mock_response = MagicMock()
        mock_response.read.return_value = (
            b'{"id": "chatcmpl-123", "choices": [{"message": {"role": "assistant", "content": "Hello from 6.1 sol!"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}'
        )
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        custom_url = "http://localhost:8000/v1/chat/completions"
        text, output, resp_id, usage = send_message(
            api_key="",
            model="gpt-6.1-sol",
            user_message="Create a cube",
            endpoint=custom_url,
        )

        self.assertEqual(text, "Hello from 6.1 sol!")
        self.assertEqual(usage.get("total_tokens"), 15)
        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.full_url, custom_url)

    @patch("gptblend.core.client.urllib.request.urlopen")
    def test_send_message_fallback_from_responses_to_chat(self, mock_urlopen):
        import urllib.error

        # First call fails with 400 messages: expected an array
        # Second call succeeds with chat completions
        err_body = b'{"error":{"code":"invalid_chat_parameter","message":"messages: expected an array","type":"invalid_request_error"}}'
        http_err = urllib.error.HTTPError(
            url="http://proxy/v1/responses",
            code=400,
            msg="Bad Request",
            hdrs={},
            fp=MagicMock(read=MagicMock(return_value=err_body)),
        )

        ok_resp = MagicMock()
        ok_resp.read.return_value = (
            b'{"id": "chatcmpl-fallback", "choices": [{"message": {"role": "assistant", "content": "Fallback works!"}}]}'
        )
        ok_resp.__enter__.return_value = ok_resp

        mock_urlopen.side_effect = [http_err, ok_resp]

        text, output, resp_id, usage = send_message(
            api_key="sk-test",
            model="gpt-6.1-sol",
            user_message="Test",
            endpoint="http://proxy/v1/responses",
        )

        self.assertEqual(text, "Fallback works!")
        self.assertEqual(mock_urlopen.call_count, 2)
        second_call_url = mock_urlopen.call_args_list[1][0][0].full_url
        self.assertEqual(second_call_url, "http://proxy/v1/chat/completions")


if __name__ == "__main__":
    unittest.main()
