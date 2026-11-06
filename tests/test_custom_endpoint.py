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
from gptblend.core.client import send_message, GPTBlendError


class DummyPrefs:
    def __init__(self, api_endpoint="", model="gpt-6-sol", custom_model=""):
        self.api_endpoint = api_endpoint
        self.model = model
        self.custom_model = custom_model


class TestCustomEndpoint(unittest.TestCase):

    def test_get_effective_endpoint_default(self):
        prefs = DummyPrefs(api_endpoint="")
        self.assertEqual(get_effective_endpoint(prefs), DEFAULT_API_URL)

        prefs = DummyPrefs(api_endpoint="   ")
        self.assertEqual(get_effective_endpoint(prefs), DEFAULT_API_URL)

    def test_get_effective_endpoint_normalizes_v1(self):
        prefs = DummyPrefs(api_endpoint="http://localhost:8000/v1")
        self.assertEqual(
            get_effective_endpoint(prefs), "http://localhost:8000/v1/responses"
        )

        prefs = DummyPrefs(api_endpoint="https://my-proxy.com/v1/")
        self.assertEqual(
            get_effective_endpoint(prefs), "https://my-proxy.com/v1/responses"
        )

    def test_get_effective_endpoint_custom_full_url(self):
        url = "http://127.0.0.1:11434/v1/responses"
        prefs = DummyPrefs(api_endpoint=url)
        self.assertEqual(get_effective_endpoint(prefs), url)

    def test_get_effective_model_preset(self):
        prefs = DummyPrefs(model="gpt-6-astra")
        self.assertEqual(get_effective_model(prefs), "gpt-6-astra")

    def test_get_effective_model_custom(self):
        prefs = DummyPrefs(model="custom", custom_model="deepseek-coder-v2")
        self.assertEqual(get_effective_model(prefs), "deepseek-coder-v2")

        prefs_empty = DummyPrefs(model="custom", custom_model="  ")
        self.assertEqual(get_effective_model(prefs_empty), "gpt-6-sol")

    def test_send_message_requires_key_for_openai(self):
        with self.assertRaises(GPTBlendError) as ctx:
            send_message(
                api_key="",
                model="gpt-6-sol",
                user_message="Hello",
                endpoint=DEFAULT_API_URL,
            )
        self.assertIn("No OpenAI API key", str(ctx.exception))

    @patch("gptblend.core.client.urllib.request.urlopen")
    def test_send_message_custom_endpoint_without_key(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"output_text": "OK"}'
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        custom_url = "http://localhost:8000/v1/responses"
        text, output, resp_id, usage = send_message(
            api_key="",
            model="custom-model",
            user_message="Hello",
            endpoint=custom_url,
        )

        self.assertEqual(text, "OK")
        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.full_url, custom_url)
        self.assertNotIn("Authorization", req.headers)

    @patch("gptblend.core.client.urllib.request.urlopen")
    def test_send_message_custom_endpoint_with_key(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"output_text": "Response from proxy"}'
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        custom_url = "https://custom-proxy.internal/v1/responses"
        text, output, resp_id, usage = send_message(
            api_key="sk-custom-secret",
            model="custom-model",
            user_message="Hello",
            endpoint=custom_url,
        )

        self.assertEqual(text, "Response from proxy")
        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.full_url, custom_url)
        self.assertEqual(req.headers.get("Authorization"), "Bearer sk-custom-secret")


if __name__ == "__main__":
    unittest.main()
