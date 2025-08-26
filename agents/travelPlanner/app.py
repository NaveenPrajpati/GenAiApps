from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
import requests


class AgentState(TypedDict):
    weather: Annotated[list, add_messages]
    location: Annotated[list, add_messages]


def getWeather(state: AgentState):
    response = requests.get("weather api")
    if response.status_code == 200:
        data = response.json()
        print("weather-", data)
        return {"weather": data}
    else:
        print("weather request failed")


def getLocation(state: AgentState):
    response = requests.get("location api")
    if response.status_code == 200:
        data = response.json()
        print("location-", data)
        return {"location": data}
    else:
        print("location request failed")


graph = StateGraph(AgentState)
graph.add_node("getWeather", getWeather)
graph.add_node("getLocation", getLocation)
