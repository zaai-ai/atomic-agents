import instructor
import openai
from pydantic import Field
from typing import List, Optional

from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseIOSchema
from atomic_agents.lib.components.system_prompt_generator import SystemPromptContextProviderBase, SystemPromptGenerator


class BlogWriterInputSchema(BaseIOSchema):
    """This schema defines the input schema for the BlogWriterAgent."""

    summary: str = Field(
        ..., description="A short summary of the content, including who is presenting and the content being discussed."
    )
    insights: List[str] = Field(
        ..., description="the best insights and ideas of the content."
    )
    quotes: List[str] = Field(
        ...,
        description="the most surprising, insightful, and/or interesting quotes of the content.",
    )
    habits: Optional[List[str]] = Field(
        None,
        description="the most practical and useful personal habits mentioned.",
    )
    facts: List[str] = Field(
        ...,
        description="the most surprising, insightful, and/or interesting valid facts about the greater world mentioned in the content.",
    )
    recommendations: List[str] = Field(
        ...,
        description="the most surprising, insightful, and/or interesting recommendations from the content.",
    )
    references: List[str] = Field(
        ...,
        description="All mentions of writing, art, tools, projects, and other sources of inspiration mentioned in the content.",
    )
    one_sentence_takeaway: str = Field(
        ..., description="The most potent takeaways and recommendations condensed into a single 20-word sentence."
    )


class BlogWriterOutputSchema(BaseIOSchema):
    """This schema defines how the article should be written."""

    introduction: str = Field(
        ..., description="A 3 paragraph introduction of explaining what the article is about and what the reader should expect from the article."
    )
    body: List[str] = Field(
        ..., description="A thorough and detailed explanation of the theme of the article based on the input."
    )
    code: List[str] = Field(
        ..., description="A simple use case example on how to code what has been explained in the article. You must include mock up data and a code block in your output."
    )
    conclusion: List[str] = Field(
        ..., description="A 3 paragraph conclusion about the article and what the reader should take away from it.",
    )


blog_writer_agent = BaseAgent(
    config=BaseAgentConfig(
        client=instructor.from_openai(openai.OpenAI()),
        model="gpt-4o-mini",
        system_prompt_generator=SystemPromptGenerator(
            background=[
                "This Assistant is an expert at writing blog posts about machine learning topics based."
            ],
            steps=[
                "Analyse the content provided thoroughly to write an exceptional article with introdution, body, code and conclusion.",
                "Adhere strictly to the provided schema when extracting information from the input content.",
                "Ensure that the output matches the field descriptions, types and constraints exactly.",
            ],
            output_instructions=[
                "Only output Markdown-compatible strings.",
                "Ensure you follow ALL these instructions when creating your output.",
            ],
        ),
        input_schema=BlogWriterInputSchema,
        output_schema=BlogWriterOutputSchema,
    )
)
