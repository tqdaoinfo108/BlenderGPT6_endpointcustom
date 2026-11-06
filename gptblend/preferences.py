import bpy
from bpy.types import AddonPreferences, Operator
from bpy.props import StringProperty, EnumProperty, IntProperty, BoolProperty

DEFAULT_API_URL = "https://api.openai.com/v1/responses"


def get_effective_endpoint(prefs=None):
    if prefs is None:
        prefs = get_preferences()
    endpoint = getattr(prefs, "api_endpoint", "").strip() if prefs else ""
    if not endpoint:
        return DEFAULT_API_URL
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/v1"):
        endpoint = f"{endpoint}/responses"
    return endpoint


def get_effective_model(prefs=None):
    if prefs is None:
        prefs = get_preferences()
    if not prefs:
        return "gpt-6-sol"
    if prefs.model == "custom":
        custom = getattr(prefs, "custom_model", "").strip()
        return custom if custom else "gpt-6-sol"
    return prefs.model


class GPTBlendResetEndpointOperator(Operator):
    bl_idname = "gptblend.reset_endpoint"
    bl_label = "Reset Endpoint"
    bl_description = "Reset API endpoint to default OpenAI endpoint"

    def execute(self, context):
        prefs = get_preferences()
        if prefs:
            prefs.api_endpoint = DEFAULT_API_URL
        return {"FINISHED"}


class GPTBlendPreferences(AddonPreferences):
    bl_idname = __package__

    api_key: StringProperty(
        name="API Key",
        description="API key used by GPT Blend. Optional if using a local/custom endpoint that does not require authentication.",
        subtype="PASSWORD",
        default="",
    )

    api_endpoint: StringProperty(
        name="API Endpoint",
        description="Custom API endpoint URL. Defaults to https://api.openai.com/v1/responses. Supports custom proxies and compatible endpoints.",
        default=DEFAULT_API_URL,
    )

    model: EnumProperty(
        name="Model",
        items=[
            ("gpt-6-astra", "GPT-6 Astra", "Current GPT-6 flagship model for demanding agentic work"),
            ("gpt-6-sol", "GPT-6 Sol", "GPT-6 model for complex coding and agentic workflows"),
            ("gpt-6-luna", "GPT-6 Luna", "GPT-6 model for efficient, high-volume work"),
            ("gpt-5.6-sol", "GPT-5.6 Sol", "GPT-5.6 flagship model"),
            ("gpt-5.6-terra", "GPT-5.6 Terra", "GPT-5.6 model balancing capability and cost"),
            ("gpt-5.6-luna", "GPT-5.6 Luna", "GPT-5.6 model for cost-sensitive, high-volume work"),
            ("custom", "Custom Model", "Use a custom model specified in the field below"),
        ],
        default="gpt-6-sol",
    )

    custom_model: StringProperty(
        name="Custom Model",
        description="Custom model identifier (e.g., deepseek-chat, gpt-4o, etc.) used when Model is set to 'Custom Model'.",
        default="",
    )

    max_tool_rounds: IntProperty(
        name="Max Agent Rounds",
        description="Maximum number of model/tool continuation rounds for one prompt.",
        default=100,
        min=1,
        max=200,
    )

    max_total_tool_calls: IntProperty(
        name="Max Total Tool Calls",
        description="Hard safety cap on the total Blender tool calls in one prompt.",
        default=150,
        min=1,
        max=500,
    )

    loop_protection: BoolProperty(
        name="Loop Protection",
        description="Stop the agent when it repeats the exact same tool call too many times.",
        default=True,
    )

    allow_destructive_operations: BoolProperty(
        name="Allow Destructive Operations",
        description="Allow GPT Blend to delete objects, apply modifiers, and join objects without an additional confirmation step.",
        default=False,
    )

    include_viewport_snapshot: BoolProperty(
        name="Send Viewport Snapshot",
        description="Include a screenshot of the active 3D Viewport with each new prompt so GPT can reason about visible layout, materials, and composition.",
        default=True,
    )

    def draw(self, context):
        layout = self.layout
        layout.label(text="GPT Blend — Connection & API", icon="URL")
        layout.prop(self, "api_key")

        endpoint_row = layout.row(align=True)
        endpoint_row.prop(self, "api_endpoint")
        endpoint_row.operator("gptblend.reset_endpoint", text="", icon="FILE_REFRESH")

        layout.prop(self, "model")
        if self.model == "custom":
            layout.prop(self, "custom_model")

        box = layout.box()
        box.label(text="Agent Controls", icon="SETTINGS")
        box.prop(self, "max_tool_rounds")
        box.prop(self, "max_total_tool_calls")
        box.prop(self, "loop_protection")
        box.prop(self, "allow_destructive_operations")
        box.prop(self, "include_viewport_snapshot")


classes = (
    GPTBlendResetEndpointOperator,
    GPTBlendPreferences,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


def get_preferences():
    try:
        return bpy.context.preferences.addons[__package__].preferences
    except Exception:
        return None
