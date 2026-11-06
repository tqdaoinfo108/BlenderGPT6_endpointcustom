import json
import time
import urllib.request
import urllib.error

from ..tools.registry import get_tools, run_tool

DEFAULT_API_URL = "https://api.openai.com/v1/responses"
API_URL = DEFAULT_API_URL
MAX_TOOL_ROUNDS = 100
MAX_TOTAL_TOOL_CALLS = 150
MAX_IDENTICAL_TOOL_CALLS = 4
MAX_REQUEST_RETRIES = 3
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

VERIFY_OBJECT_ARGUMENT = {
    "create_object": "name",
    "transform_object": "name",
    "rename_object": "new_name",
    "duplicate_object": "new_name",
    "set_material": "object_name",
    "set_object_color": "name",
    "move_object_delta": "name",
    "rotate_object_delta": "name",
    "set_object_dimensions": "name",
    "apply_object_scale": "name",
    "set_procedural_texture": "object_name",
    "create_light": "name",
    "set_light_settings": "light_name",
    "create_camera": "name",
    "set_camera_settings": "camera_name",
    "animate_object_transform": "object_name",
    "animate_object_visibility": "object_name",
    "set_animation_cycles": "object_name",
    "set_camera_depth_of_field": "camera_name",
    "add_tracking_constraint": "object_name",
    "add_copy_transforms_constraint": "object_name",
}


class GPTBlendError(Exception):
    pass


def get_chat_tools():
    """Convert flat tool definitions to OpenAI Chat Completions function tool format."""
    raw_tools = get_tools()
    chat_tools = []
    for tool in raw_tools:
        if "function" in tool:
            chat_tools.append(tool)
        else:
            chat_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                },
            })
    return chat_tools


def _request(api_key, payload, endpoint=None):
    url = (endpoint or "").strip() or DEFAULT_API_URL
    last_error = None

    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    for attempt in range(MAX_REQUEST_RETRIES + 1):
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = GPTBlendError(f"API error ({exc.code}): {body}")
            if exc.code not in RETRYABLE_STATUS_CODES or attempt >= MAX_REQUEST_RETRIES:
                raise last_error from exc
            time.sleep(min(2 ** attempt, 8))
        except urllib.error.URLError as exc:
            last_error = GPTBlendError(f"Network error: {exc.reason}")
            if attempt >= MAX_REQUEST_RETRIES:
                raise last_error from exc
            time.sleep(min(2 ** attempt, 8))
        except Exception as exc:
            raise GPTBlendError(f"Request failed: {exc}") from exc

    raise last_error or GPTBlendError("Request failed after retries.")


def _extract_text(data):
    output = data.get("output_text")
    if output:
        return output

    chunks = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                chunks.append(content["text"])
    return "\n".join(chunks)


def _send_message_chat(
    api_key,
    model,
    user_message,
    instructions,
    endpoint,
    max_tool_rounds=MAX_TOOL_ROUNDS,
    max_total_tool_calls=MAX_TOTAL_TOOL_CALLS,
    loop_protection=True,
    allow_destructive_operations=False,
    tool_executor=None,
    progress_callback=None,
    cancel_event=None,
    viewport_image_data_url=None,
):
    """Client loop for OpenAI Chat Completions API (/v1/chat/completions)."""
    def progress(message):
        if progress_callback:
            try:
                progress_callback(message)
            except Exception:
                pass

    def cancelled():
        return bool(cancel_event and cancel_event.is_set())

    messages = [
        {"role": "system", "content": instructions},
    ]

    if viewport_image_data_url:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_message},
                {
                    "type": "image_url",
                    "image_url": {"url": viewport_image_data_url, "detail": "auto"},
                },
            ],
        })
    else:
        messages.append({"role": "user", "content": user_message})

    payload = {
        "model": model,
        "messages": messages,
        "tools": get_chat_tools(),
        "tool_choice": "auto",
    }

    tool_call_count = 0
    last_signature = None
    identical_call_count = 0

    for round_index in range(max_tool_rounds):
        if cancelled():
            raise GPTBlendError("GPT Blend task cancelled.")
        progress(f"Thinking (round {round_index + 1}/{max_tool_rounds})")

        if tool_call_count >= max_total_tool_calls:
            raise GPTBlendError(
                f"GPT Blend stopped after {max_total_tool_calls} total tool calls to prevent runaway execution."
            )

        progress("Waiting for model response...")
        data = _request(api_key, payload, endpoint=endpoint)

        choices = data.get("choices", [])
        if not choices:
            err = data.get("error")
            if err:
                msg = err.get("message") if isinstance(err, dict) else str(err)
                raise GPTBlendError(f"API error: {msg}")
            return "", [], data.get("id"), data.get("usage") or {}

        choice = choices[0]
        msg = choice.get("message", {})
        assistant_text = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            usage = data.get("usage") or {}
            norm_usage = {
                "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
                "output_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
                "total_tokens": usage.get("total_tokens", 0),
            }
            return assistant_text, [msg], data.get("id"), norm_usage

        messages.append(msg)

        for call in tool_calls:
            tool_call_count += 1
            if tool_call_count > max_total_tool_calls:
                raise GPTBlendError(f"GPT Blend stopped after {max_total_tool_calls} total tool calls.")

            call_id = call.get("id", f"call_{tool_call_count}")
            fn = call.get("function", {})
            name = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")

            try:
                arguments = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except json.JSONDecodeError as exc:
                result = {"ok": False, "message": f"Invalid tool arguments: {exc}"}
            else:
                signature = json.dumps({"name": name, "arguments": arguments}, sort_keys=True, separators=(",", ":"))
                if signature == last_signature:
                    identical_call_count += 1
                else:
                    identical_call_count = 1
                    last_signature = signature

                if loop_protection and identical_call_count > MAX_IDENTICAL_TOOL_CALLS:
                    raise GPTBlendError(
                        f"GPT Blend stopped because the same tool call repeated {MAX_IDENTICAL_TOOL_CALLS} times."
                    )

                if (
                    name in {"delete_object", "apply_modifier", "join_objects", "merge_selected_vertices", "dissolve_selected"}
                    and not allow_destructive_operations
                ):
                    result = {
                        "ok": False,
                        "confirmation_required": True,
                        "message": (
                            f"'{name}' is a destructive operation and is disabled "
                            "until destructive operations are enabled in GPT Blend Preferences."
                        ),
                    }
                else:
                    if cancelled():
                        raise GPTBlendError("GPT Blend task cancelled.")
                    progress(f"Executing {name}...")
                    try:
                        if tool_executor is not None:
                            result = tool_executor(name, arguments)
                        else:
                            result = run_tool(name, arguments)

                        verify_key = VERIFY_OBJECT_ARGUMENT.get(name)
                        verification_tool = None
                        verification_args = None

                        if name in {
                            "set_animation_timing",
                            "set_render_settings",
                            "set_render_output",
                            "add_scene_marker",
                            "remove_scene_marker",
                            "set_world_background",
                        }:
                            verification_tool = "inspect_scene"
                            verification_args = {}
                        elif verify_key:
                            verification_tool = (
                                "inspect_animation"
                                if name in {"animate_object_transform", "animate_object_visibility"}
                                else "inspect_object"
                            )
                            verification_args = (
                                {"object_name": arguments.get(verify_key)}
                                if verification_tool == "inspect_animation"
                                else {"name": arguments.get(verify_key)}
                            )

                        if (
                            result.get("ok")
                            and verification_tool
                            and tool_executor is not None
                            and tool_call_count < max_total_tool_calls
                            and (not verify_key or arguments.get(verify_key))
                        ):
                            tool_call_count += 1
                            progress("Verifying the Blender change...")
                            verification = tool_executor(verification_tool, verification_args)
                            if verification.get("ok"):
                                result["verification"] = (
                                    verification
                                    if verification_tool in {"inspect_scene", "inspect_animation"}
                                    else verification.get("object")
                                )
                    except Exception as exc:
                        result = {"ok": False, "message": f"Tool execution error: {exc}"}

            try:
                serialized_result = json.dumps(result)
            except (TypeError, ValueError) as exc:
                serialized_result = json.dumps({
                    "ok": bool(result.get("ok")),
                    "message": str(result.get("message", "Tool result could not be serialized.")),
                    "serialization_error": str(exc),
                })

            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": serialized_result,
            })

        payload["messages"] = messages

    raise GPTBlendError(f"GPT Blend reached the configured maximum of {max_tool_rounds} agent rounds.")


def _send_message_responses(
    api_key,
    model,
    user_message,
    instructions,
    endpoint,
    history=None,
    previous_response_id=None,
    max_tool_rounds=MAX_TOOL_ROUNDS,
    max_total_tool_calls=MAX_TOTAL_TOOL_CALLS,
    loop_protection=True,
    allow_destructive_operations=False,
    tool_executor=None,
    progress_callback=None,
    cancel_event=None,
    viewport_image_data_url=None,
):
    """Client loop for OpenAI Responses API (/v1/responses)."""
    input_items = []
    if history:
        input_items.extend(history)

    user_content = [{"type": "input_text", "text": user_message}]
    if viewport_image_data_url:
        user_content.append({
            "type": "input_image",
            "image_url": viewport_image_data_url,
            "detail": "auto",
        })
    input_items.append({"role": "user", "content": user_content})

    payload = {
        "model": model,
        "instructions": instructions,
        "input": input_items,
        "tools": get_tools(),
        "parallel_tool_calls": False,
    }

    if previous_response_id:
        payload["previous_response_id"] = previous_response_id

    tool_call_count = 0
    last_signature = None
    identical_call_count = 0

    def progress(message):
        if progress_callback:
            try:
                progress_callback(message)
            except Exception:
                pass

    def cancelled():
        return bool(cancel_event and cancel_event.is_set())

    for round_index in range(max_tool_rounds):
        if cancelled():
            raise GPTBlendError("GPT Blend task cancelled.")
        progress(f"Thinking (round {round_index + 1}/{max_tool_rounds})")

        if tool_call_count >= max_total_tool_calls:
            raise GPTBlendError(
                f"GPT Blend stopped after {max_total_tool_calls} total tool calls to prevent runaway execution."
            )

        progress("Waiting for model response...")
        data = _request(api_key, payload, endpoint=endpoint)
        tool_calls = [item for item in data.get("output", []) if item.get("type") == "function_call"]

        if not tool_calls:
            return _extract_text(data), data.get("output", []), data.get("id"), data.get("usage") or {}

        tool_outputs = []
        for call in tool_calls:
            tool_call_count += 1
            if tool_call_count > max_total_tool_calls:
                raise GPTBlendError(f"GPT Blend stopped after {max_total_tool_calls} total tool calls.")

            try:
                arguments = json.loads(call.get("arguments", "{}"))
            except json.JSONDecodeError as exc:
                result = {"ok": False, "message": f"Invalid tool arguments: {exc}"}
            else:
                signature = json.dumps(
                    {"name": call.get("name"), "arguments": arguments},
                    sort_keys=True,
                    separators=(",", ":"),
                )

                if signature == last_signature:
                    identical_call_count += 1
                else:
                    identical_call_count = 1
                    last_signature = signature

                if loop_protection and identical_call_count > MAX_IDENTICAL_TOOL_CALLS:
                    raise GPTBlendError(
                        f"GPT Blend stopped because the same tool call repeated "
                        f"{MAX_IDENTICAL_TOOL_CALLS} times."
                    )

                if (
                    call.get("name") in {"delete_object", "apply_modifier", "join_objects", "merge_selected_vertices", "dissolve_selected"}
                    and not allow_destructive_operations
                ):
                    result = {
                        "ok": False,
                        "confirmation_required": True,
                        "message": (
                            f"'{call.get('name')}' is a destructive operation and is disabled "
                            "until destructive operations are enabled in GPT Blend Preferences."
                        ),
                    }
                else:
                    if cancelled():
                        raise GPTBlendError("GPT Blend task cancelled.")
                    progress(f"Executing {call.get('name')}...")
                    try:
                        if tool_executor is not None:
                            result = tool_executor(call.get("name"), arguments)
                        else:
                            result = run_tool(call.get("name"), arguments)

                        verify_key = VERIFY_OBJECT_ARGUMENT.get(call.get("name"))
                        verification_tool = None
                        verification_args = None

                        if call.get("name") in {
                            "set_animation_timing",
                            "set_render_settings",
                            "set_render_output",
                            "add_scene_marker",
                            "remove_scene_marker",
                            "set_world_background",
                        }:
                            verification_tool = "inspect_scene"
                            verification_args = {}
                        elif verify_key:
                            verification_tool = (
                                "inspect_animation"
                                if call.get("name") in {"animate_object_transform", "animate_object_visibility"}
                                else "inspect_object"
                            )
                            verification_args = (
                                {"object_name": arguments.get(verify_key)}
                                if verification_tool == "inspect_animation"
                                else {"name": arguments.get(verify_key)}
                            )

                        if (
                            result.get("ok")
                            and verification_tool
                            and tool_executor is not None
                            and tool_call_count < max_total_tool_calls
                            and (not verify_key or arguments.get(verify_key))
                        ):
                            tool_call_count += 1
                            progress("Verifying the Blender change...")
                            verification = tool_executor(
                                verification_tool,
                                verification_args,
                            )
                            if verification.get("ok"):
                                result["verification"] = (
                                    verification
                                    if verification_tool in {"inspect_scene", "inspect_animation"}
                                    else verification.get("object")
                                )
                    except Exception as exc:
                        result = {"ok": False, "message": f"Tool execution error: {exc}"}

            try:
                serialized_result = json.dumps(result)
            except (TypeError, ValueError) as exc:
                serialized_result = json.dumps({
                    "ok": bool(result.get("ok")),
                    "message": str(result.get("message", "Tool result could not be serialized.")),
                    "serialization_error": str(exc),
                })
            tool_outputs.append({
                "type": "function_call_output",
                "call_id": call.get("call_id"),
                "output": serialized_result,
            })

        payload = {
            "model": model,
            "instructions": instructions,
            "previous_response_id": data.get("id"),
            "input": tool_outputs,
            "tools": get_tools(),
            "parallel_tool_calls": False,
        }

    raise GPTBlendError(
        f"GPT Blend reached the configured maximum of {max_tool_rounds} agent rounds."
    )


def send_message(
    api_key,
    model,
    user_message,
    context_text="",
    endpoint=None,
    history=None,
    previous_response_id=None,
    max_tool_rounds=MAX_TOOL_ROUNDS,
    max_total_tool_calls=MAX_TOTAL_TOOL_CALLS,
    loop_protection=True,
    allow_destructive_operations=False,
    tool_executor=None,
    progress_callback=None,
    cancel_event=None,
    viewport_image_data_url=None,
):
    target_url = (endpoint or "").strip() or DEFAULT_API_URL
    is_openai_default = "openai.com" in target_url
    if not api_key and is_openai_default:
        raise GPTBlendError("No OpenAI API key configured.")

    max_tool_rounds = max(1, min(int(max_tool_rounds), 200))
    max_total_tool_calls = max(1, min(int(max_total_tool_calls), 500))

    instructions = (
        "You are GPT Blend, an interactive AI assistant inside Blender. "
        "You can inspect and modify the Blender workspace using the provided Blender tools. "
        "Prefer tools over writing bpy Python code. "
        "Do not claim an action happened unless a tool returned success. "
        "When a task requires multiple steps, perform those steps with tools and verify important results. "
        "Use inspect_scene or inspect_object when object names, transforms, modifiers, materials, "
        "or collections are uncertain. "
        "Do not delete objects unless the user explicitly requests deletion. "
        "Do not apply modifiers, join objects, or change scene structure unless it helps fulfill the user's request. "
        "Keep changes scoped to the user's request and avoid unnecessary edits. "
        "When the user asks for visible color, materials, or textures, actually create/apply the material with the material tools. "
        "When a procedural texture is requested, use set_procedural_texture rather than only changing viewport color. "
        "When the user expects to see the material or texture in the 3D viewport, use set_viewport_shading with MATERIAL as needed. "
        "When the user asks for an animation, use the animation tools to set scene timing and create explicit transform keyframes instead of only moving the object once. "
        "For camera, light, or object animation, use animate_object_transform and verify the authored action afterward. "
        "For object appear/disappear timing, use animate_object_visibility rather than repeatedly editing visibility manually. "
        "Use set_animation_cycles when a motion should loop. "
        "Use tracking or Copy Transforms constraints when the relationship should remain procedural rather than baking every relationship into keyframes. "
        "Use camera settings tools to adjust projection, lens, clipping, and sensor shift; use depth of field tools for focus-object or cinematic focus requests. "
        "Use light settings tools to tune existing lights instead of creating duplicate lights when the user asks to change lighting. "
        "After important material, texture, camera, lighting, or geometry changes, verify the result with an inspection tool before claiming success. "
        "Use set_render_output when the user specifies an output folder, file format, or transparency. "
        "Use the provided viewport image as visual evidence when present; distinguish what is visible there from what must be verified through Blender tools.\n\n"
        f"CURRENT BLENDER CONTEXT:\n{context_text or 'No context available.'}"
    )

    # Determine whether endpoint expects Chat Completions or Responses API
    # 1. If URL has "chat/completions" -> Chat Completions
    # 2. If URL has "responses" -> Responses API
    # 3. If custom URL (not official api.openai.com/v1/responses) -> default to Chat Completions
    url_lower = target_url.lower()
    is_chat = "chat/completions" in url_lower or ("responses" not in url_lower and url_lower != DEFAULT_API_URL.lower())

    if is_chat:
        # Normalize endpoint URL for chat completions if needed
        chat_url = target_url
        if chat_url.rstrip("/").endswith("/v1"):
            chat_url = f"{chat_url.rstrip('/')}/chat/completions"
        elif not chat_url.endswith("/chat/completions") and not chat_url.endswith("/responses"):
            # If path doesn't end with chat/completions
            if "/" in chat_url.split("://")[-1]:
                if not chat_url.endswith("/completions"):
                    chat_url = f"{chat_url.rstrip('/')}/chat/completions"
            else:
                chat_url = f"{chat_url.rstrip('/')}/v1/chat/completions"

        return _send_message_chat(
            api_key=api_key,
            model=model,
            user_message=user_message,
            instructions=instructions,
            endpoint=chat_url,
            max_tool_rounds=max_tool_rounds,
            max_total_tool_calls=max_total_tool_calls,
            loop_protection=loop_protection,
            allow_destructive_operations=allow_destructive_operations,
            tool_executor=tool_executor,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            viewport_image_data_url=viewport_image_data_url,
        )

    # Responses API with auto-fallback to Chat Completions on 400 invalid_chat_parameter
    try:
        return _send_message_responses(
            api_key=api_key,
            model=model,
            user_message=user_message,
            instructions=instructions,
            endpoint=target_url,
            history=history,
            previous_response_id=previous_response_id,
            max_tool_rounds=max_tool_rounds,
            max_total_tool_calls=max_total_tool_calls,
            loop_protection=loop_protection,
            allow_destructive_operations=allow_destructive_operations,
            tool_executor=tool_executor,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            viewport_image_data_url=viewport_image_data_url,
        )
    except GPTBlendError as exc:
        err_str = str(exc).lower()
        if "messages: expected an array" in err_str or "invalid_chat_parameter" in err_str or "missing required parameter: 'messages'" in err_str:
            fallback_url = target_url.replace("/responses", "/chat/completions")
            if fallback_url == target_url:
                fallback_url = f"{target_url.rstrip('/')}/chat/completions"
            return _send_message_chat(
                api_key=api_key,
                model=model,
                user_message=user_message,
                instructions=instructions,
                endpoint=fallback_url,
                max_tool_rounds=max_tool_rounds,
                max_total_tool_calls=max_total_tool_calls,
                loop_protection=loop_protection,
                allow_destructive_operations=allow_destructive_operations,
                tool_executor=tool_executor,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
                viewport_image_data_url=viewport_image_data_url,
            )
        raise
