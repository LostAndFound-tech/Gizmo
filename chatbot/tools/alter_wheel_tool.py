"""
tools/alter_wheel_tool.py
Gizmo tool that talks to the Alter Wheel API on the Pi.
Called automatically when switch_host fires.
"""

import os
import httpx
from tools.base_tool import BaseTool, ToolResult

WHEEL_URL = os.getenv("ALTER_WHEEL_URL", "http://raspberrypi.local:8765")


class AlterWheelTool(BaseTool):
    @property
    def name(self) -> str:
        return "alter_wheel"

    @property
    def description(self) -> str:
        return (
            "Updates the physical Alter Wheel LED display. "
            "Call after a host switch to reflect the change visually. "
            "Args: action (str) — 'switch', 'add_fronter', or 'remove_fronter'. "
            "new_host (str) — for switch action. "
            "name (str) — for add/remove actions. "
            "staying_fronters (list) — optional, who stays co-fronting on switch."
        )

    async def run(
        self,
        action: str = "switch",
        new_host: str = "",
        name: str = "",
        staying_fronters: list = None,
        **kwargs,
    ) -> ToolResult:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                if action == "switch":
                    resp = await client.post(
                        f"{WHEEL_URL}/switch",
                        json={
                            "new_host": new_host,
                            "staying_fronters": staying_fronters or [],
                        }
                    )
                elif action == "add_fronter":
                    resp = await client.post(
                        f"{WHEEL_URL}/add_fronter",
                        json={"name": name}
                    )
                elif action == "remove_fronter":
                    resp = await client.post(
                        f"{WHEEL_URL}/remove_fronter",
                        json={"name": name}
                    )
                elif action == "status":
                    resp = await client.get(f"{WHEEL_URL}/status")
                else:
                    return ToolResult(success=False, output=f"Unknown action: {action}")

                data = resp.json()
                if resp.status_code == 200:
                    return ToolResult(
                        success=True,
                        output=f"Wheel updated: {data}",
                        data=data,
                    )
                else:
                    return ToolResult(
                        success=False,
                        output=f"Wheel error: {data.get('error', 'unknown')}",
                    )

        except httpx.ConnectError:
            return ToolResult(
                success=False,
                output="Alter Wheel not reachable — is the Pi running?",
            )
        except Exception as e:
            return ToolResult(success=False, output=f"Wheel error: {e}")
