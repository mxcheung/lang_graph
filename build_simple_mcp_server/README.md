
# Lab Overview
In this Code Lab, you will build a simple but compliant Model Context Protocol (MCP) server from the ground up using Python and the FastAPI framework. You will learn how to structure MCP messages, handle discovery and execution requests, and securely expose custom tools to AI agents.

##
The first stage of the MCP lifecycle is Tool Registration and Discovery.

In this phase, an AI agent ask your server what tools it can use. You'll implement a handler for the discovery verb that responds with a list of all available tools — including their names, descriptions, and required parameters.

This allows the agent to learn how to use your tools dynamically.
