from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import SerperDevTool



@CrewBase
class EthicsFANG():
  """Ethics Council crew"""

  agents_config = "config/agents.yaml"
  tasks_config = "config/tasks.yaml"

  llm = LLM(model="gpt-4o-mini",
          temperature=0.3,
          max_completion_tokens=1000,
          max_tokens=5000)

  # Create the web search tool
  web_search_tool = SerperDevTool()


  @agent
  def research_analyst(self) -> Agent:
    return Agent(
      config=self.agents_config['research_analyst'], # type: ignore[index]
      verbose=True,
      tools=[self.web_search_tool],
      llm=self.llm
    )

  @agent
  def ethical_agent(self) -> Agent:
    return Agent(
      config=self.agents_config['ethical_agent'], # type: ignore[index]
      verbose=True,
      llm=self.llm
    )
  
  @agent
  def editor_agent(self) -> Agent:
    return Agent(
      config=self.agents_config['editor_agent'], # type: ignore[index]
      verbose=True,
      llm=self.llm
    )


  @task
  def ethical_outline_task(self) -> Task:
    return Task(
      config=self.tasks_config['ethical_outline_task'] # type: ignore[index]
    )

  @task
  def research_terms_task(self) -> Task:
    return Task(
      config=self.tasks_config['research_terms_task'] # type: ignore[index]
    )
  
  @task
  def research_vibe_task(self) -> Task:
    return Task(
      config=self.tasks_config['research_vibe_task'] # type: ignore[index]
    )
  
  @task
  def ethical_impact_task(self) -> Task:
    return Task(
      config=self.tasks_config['ethical_impact_task'] # type: ignore[index]
    )
  
  @task
  def editor_task(self) -> Task:
    return Task(
      config=self.tasks_config['editor_task'] # type: ignore[index]
    )
  
  
  @crew
  def crew(self) -> Crew:
    return Crew(
      agents=[
        self.research_analyst(),
        self.ethical_agent(),
        self.editor_agent()
      ],
      tasks=[
        self.ethical_outline_task(),
        self.research_terms_task(),
        self.research_vibe_task(),
        self.ethical_impact_task(),
        self.editor_task()
      ],
      process=Process.sequential,
      verbose=True,
    )
