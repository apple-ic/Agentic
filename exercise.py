from crewai import Agent
from crewai import Task
from crewai import LLM
from dotenv import load_dotenv
from crewai import Crew, Process
from crewai_tools import SerperDevTool
import os

load_dotenv()

llm = LLM(model="gpt-4o-mini")

# Create the web search tool
web_search_tool = SerperDevTool()


# The client's domain(url) and the project description

customer_domain="crewai.com"
project_description="""CrewAI, a leading provider of multi-agent systems, aims to revolutionize marketing automation for its enterprise clients.
This project involves developing an innovative marketing strategy to showcase CrewAI's advanced AI-driven solutions,
emphasizing ease of use, scalability, and integration capabilities.
The campaign will target tech-savvy decision-makers in medium to large enterprises, highlighting success stories and
the transformative potential of CrewAI's platform."""

# Exercise 2: Define Your Agents

research_analyst = Agent(
        role="""Domain Researcher""",
        backstory="""Expert in online research, trend analysis, and data gathering across industries. Skilled at summarizing findings into actionable insights.""",
        goal="""Conduct deep research on a given topic, market, or audience and return a structured brief with key insights, stats, and references.""",
        verbose=True,
        tools=[web_search_tool],
        llm=llm
    )

strategist_agent =  Agent(
    role="""Marketing Strategist""",
    backstory="""A seasoned expert in digital marketing, audience segmentation, and branding.
                Adept at translating insights into actionable strategy to enhance brand positioning.""",
    goal="""Develop clear and compelling communication and marketing strategies based on research briefs.
        Define tone, suitable channels, target audience, and key messaging.""",
    verbose=True,
    allow_delegation=True,
    llm=llm
)

creative_director = Agent(
    role="Campaign Ideator",
    goal="Propose 2-3 unique creative campaign ideas aligned with the strategy and tailored to the audience.",
    backstory="Experienced in advertising and brand storytelling. Thinks conceptually to create campaign hooks and slogans.",
    verbose=True,
    llm=llm
)

content_writer = Agent(
    role="Content Writer",
    goal="Write clear, engaging content based on selected campaign ideas (e.g., blog posts, ad copy, social captions)",
    backstory="Experienced in writing and brand storytelling. Thinks conceptually to create campaign hooks and slogans.",
    verbose=True,
    llm=llm
)

editor_agent = Agent(
    role="Content Reviewer",
    backstory="An editorial expert skilled in proofreading, grammar checking, tone alignment, and ensuring brand consistency.",
    goal="Review the content for clarity, grammar, tone, and strategy alignment. Return final approved versions with notes if needed.",
    verbose=True,
    llm=llm
)


#  Exercise 3: Define the Tasks¶

research_task = Task(
    agent=research_analyst,
    description=f"Research the customer landscape and competitors for {project_description} in {customer_domain}. "
                f"Include emerging trends, audience demographics, and competitor positioning.",
    expected_output="A detailed research brief including statistics, audience personas, competitor analysis, and market trends."
)

strategist_task = Task(
    agent=strategist_agent,
    description=f"Develop a comprehensive communication strategy for {project_description}, targeting {customer_domain}. Base the strategy on the research findings, focusing on tone, messaging pillars, and key distribution channels.",
    expected_output="A strategy document detailing audience segmentation, brand tone, message framework, and recommended communication channels."
)


campaign_ideation_task = Task(
    agent=creative_director,
    description=f"""Based on the strategic brief, propose 2–3 unique and creative campaign ideas for {project_description}.
                 Each idea should include a campaign name, central slogan, and a short description.
                 Ensure alignment with the target audience and communication goals in the {customer_domain}. """,
    expected_output="2–3 creative campaign ideas including name, hook, and brief description."
)

content_writer_task = Task(
    agent=content_writer,
    description=f"""Write content for the selected campaign idea supporting {project_description} and designed for {customer_domain}. 
                    Include copy for one blog post, one social media post, and one ad snippet. """,
    expected_output="A polished set of written assets (blog post, social caption, ad snippet) aligned with the campaign tone and messaging."
)

editor_task = Task(
    agent=editor_agent,
    description=f"Review all content created for {project_description}, targeting {customer_domain}. Check for grammatical accuracy, brand voice alignment, clarity, and adherence to the strategy.",
    expected_output="A finalised, edited version of the content with comments and approval status.",
    output_file=f"output/{customer_domain}_content.txt"
)


# Crew

agent_list=[
    research_analyst,
    strategist_agent,
    creative_director,
    content_writer,
    editor_agent
]

task_list = [
    research_task,
    strategist_task,
    campaign_ideation_task,
    content_writer_task,
    editor_task
]


marketing_crew = Crew(
    agents=agent_list,
    tasks=task_list,
    process=Process.sequential, # or Process.hierarchical
    verbose=True
)

result = marketing_crew.kickoff(inputs={"user_input": "???"})
