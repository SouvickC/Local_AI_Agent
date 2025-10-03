## Define some of the tools we need


from mcp.server.fastmcp import FastMCP

mcp =FastMCP("math") ## server name

@mcp.tool()
def add(*args:int)->int:
    """addition of numbers"""
    total = 0
    for num in args:
        total += num
    return total


@mcp.tool()
def multiply(*args:int)->int:
    """Multiplication of numbers"""
    total =1
    for num in args:
        total *= num
    return total


if __name__ == "__main__":
    mcp.run(transport="stdio") ## standard input output transport

    ## run in the local but http use like api