from os import environ as env

import aiohttp
from dotenv import load_dotenv
from llama_index.core.tools import FunctionTool

load_dotenv()


async def get_tle(norad_id: int) -> str:
    """Fetch the latest TLE data for a satellite given its NORAD catalog id."""
    url = f"https://celestrak.com/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=tle"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                return f"Error fetching TLE data: {response.status}"
            return (await response.text()).strip()


async def get_satellite_position(
    norad_id: int,
    observer_lat: float,
    observer_lon: float,
    observer_alt: float,
) -> str:
    """Return the current position of a satellite for a given observer location."""
    api_key = env.get("N2YO_API_KEY", "")
    url = (
        f"https://api.n2yo.com/rest/v1/satellite/positions/{norad_id}/"
        f"{observer_lat}/{observer_lon}/{observer_alt}/2/&apiKey={api_key}"
    )
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                return f"Error fetching satellite position: {response.status}"
            data = await response.json()

    positions = data.get("positions") or []
    if not positions:
        return "No position data available."
    pos = positions[0]
    return (
        f"Latitude: {pos['satlatitude']}, "
        f"Longitude: {pos['satlongitude']}, "
        f"Altitude: {pos['sataltitude']} km"
    )


get_tle_tool = FunctionTool.from_defaults(async_fn=get_tle, name="get_tle")
get_satellite_position_tool = FunctionTool.from_defaults(
    async_fn=get_satellite_position,
    name="get_satellite_position",
)
