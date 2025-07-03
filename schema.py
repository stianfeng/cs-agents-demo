from typing import Literal, NotRequired, List, Dict, Any, TypedDict
from pydantic import BaseModel, Field, NonNegativeInt

class UserInput(BaseModel):
    """User input to the the assistant"""

    thread_id: str = Field(description="Thread ID to persist and continue a multi-turn conversation.")
    message: str = Field(description="User input to the assistant.")
    
class ToolCall(TypedDict):
    """Represents a request to call a tool."""

    name: str
    args: dict[str, Any]
    id: str | None
    type: NotRequired[Literal["tool_call"]]

class ChatMessage(BaseModel):
    """Message in a chat"""
    
    type: Literal["human", "ai", "tool"] = Field(description="Role of the message.")
    content: str = Field(description="Content of the message.")
    tool_calls: List[ToolCall] = Field(description="Tool calls in the message.", default=[])
    tool_call_id: str | None = Field(description="Tool call this message is responding to.", default=None)
    response_metadata: Dict[str, Any] = Field(description="Response metadata.", default={})

class MobilePlan(BaseModel):
    """Mobile plan details"""
    
    plan_name: str = Field(description="Name of the mobile plan.")
    price_monthly: float = Field(description="Monthly price of the mobile plan.")
    plan_type: Literal["SIM only", "Phone"] = Field(description="Whether the plan is SIM only or includes a phone.")
    contract_period: NonNegativeInt = Field(description="Contract period in months.")
    local_data: NonNegativeInt = Field(description="Local data allowance in GB.")
    roam_data: NonNegativeInt = Field(description="Roaming data allowance in GB.")
    roam_data_region: Literal["Asia", "Worldwide", None] = Field(description="Roaming data region.")
    talktime: NonNegativeInt = Field(description="Talk time allowance in minutes.")
    sms: NonNegativeInt = Field(description="SMS allowance.")
    caller_id: bool = Field(description="Whether caller ID is included in the plan.")
    
class MobilePlanRequest(BaseModel):
    """User desired mobile plan details"""
    
    price_monthly: float = Field(description="Monthly price of the mobile plan.", default=128.0)
    plan_type: Literal["SIM only", "Phone"] = Field(description="Whether the plan is SIM only or includes a phone.", default="Phone")
    contract_period: NonNegativeInt = Field(description="Contract period in months.", default=24)
    local_data: NonNegativeInt = Field(description="Local data allowance in GB.", default=160)
    roam_data: NonNegativeInt = Field(description="Roaming data allowance in GB.", default=2)
    roam_data_region: Literal["Asia", "Worldwide", None] = Field(description="Roaming data region.", default="Worldwide")
    talktime: NonNegativeInt = Field(description="Talk time allowance in minutes.", default=800)
    sms: NonNegativeInt = Field(description="SMS allowance.", default=800)
    caller_id: bool = Field(description="Whether caller ID is included in the plan.", default=False)
    