import jupyter_client
from llama_index.core.tools import FunctionTool


def _run_in_kernel(code: str, timeout: int = 60) -> dict:
    """Execute Python in an isolated Jupyter kernel and capture stdout/stderr/result."""
    km = jupyter_client.KernelManager()
    stdout_output: list[str] = []
    stderr_output: list[str] = []
    execution_result: str | None = None

    try:
        km.start_kernel()
        client = km.client()
        client.start_channels()
        client.wait_for_ready(timeout=timeout)

        try:
            client.execute(code)
            reply = client.get_shell_msg(timeout=timeout)

            if reply and reply["msg_type"] == "execute_reply":
                status = reply["content"]["status"]
                if status == "ok":
                    while True:
                        try:
                            msg = client.get_iopub_msg(timeout=1)
                        except TimeoutError:
                            break
                        msg_type = msg["msg_type"]
                        content = msg["content"]
                        if msg_type == "stream":
                            target = stdout_output if content["name"] == "stdout" else stderr_output
                            target.append(content["text"])
                        elif msg_type in ("display_data", "execute_result"):
                            data = content.get("data", {})
                            if "text/plain" in data:
                                execution_result = data["text/plain"]
                        elif msg_type == "status" and content["execution_state"] == "idle":
                            break
                elif status == "error":
                    stderr_output.append(f"{reply['content'].get('ename')}: {reply['content'].get('evalue')}\n")
                    stderr_output.extend(reply["content"].get("traceback", []))
            else:
                stderr_output.append(f"No valid execution reply: {reply}")

        except TimeoutError:
            stderr_output.append(f"Code execution exceeded the {timeout}s timeout.")
        except Exception as e:
            stderr_output.append(f"Kernel interaction error: {e}")
        finally:
            client.stop_channels()

    except Exception as e:
        stderr_output.append(f"Failed to start Jupyter kernel: {e}")
    finally:
        if km.is_alive():
            km.shutdown_kernel(now=True)

    return {
        "stdout": "".join(stdout_output).strip(),
        "stderr": "".join(stderr_output).strip(),
        "result": execution_result.strip() if execution_result else None,
    }


def code_interpreter(code: str, timeout: int = 60) -> dict:
    """Execute Python code in an isolated sandbox kernel (Python 3.13).

    Args:
        code: The Python source to execute.
        timeout: Max seconds to wait for execution.

    Returns:
        A dict with keys ``stdout``, ``stderr`` and ``result``.
    """
    return _run_in_kernel(code, timeout=timeout)


code_interpreter_tool = FunctionTool.from_defaults(
    fn=code_interpreter,
    name="code_interpreter",
)
