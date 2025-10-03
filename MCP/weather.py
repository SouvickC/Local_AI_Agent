

from mcp.server.fastmcp import FastMCP

mcp =FastMCP("weather") ## server name


@mcp.tool()

async def get_weather(location:str)->str:
    """get the weather location
    """
    return "Raining and dull weather here"


if __name__=="__main__":
    mcp.run(transport="streamable-http") ## standard input output transport