import os
import streamlit as st
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
import google.generativeai as genai
import time
import typing as t
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Set up the page configuration
st.set_page_config(
    page_title="Daily Planner & Reminder Suite",
    page_icon="📅",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        max-width: 1200px;
        padding: 2rem;
    }
    .stTextArea textarea {
        min-height: 150px !important;
    }
    .result-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        border-left: 5px solid #4e73df;
    }
    .agent-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .agent-selector {
        background-color: #f7f9fc;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e1e8ed;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📅 Daily Planner & Reminder Suite")
st.markdown("""
Welcome to your personal productivity suite! Choose from two helpful agents:

- **📅 Daily Planner**: Create comprehensive daily plans and schedules
- **⏰ Reminder Agent**: Generate intelligent reminder systems and task management

Select an agent below to get started!
""")

########## Gemini LLM wrapper (direct Gemini API via google.generativeai) ##########

class GeminiLLM:
    """
    Minimal wrapper around google.generativeai to expose a simple synchronous .call(prompt, **kwargs)
    method which returns generated text. This attempts multiple genai call styles to be
    resilient to SDK differences.
    """
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash", temperature: float = 0.7, max_output_tokens: int = 512):
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not provided")
        genai.configure(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def _extract_text_from_response(self, resp) -> str:
        """
        Attempt to pull a sensible text string from common Gemini SDK response shapes.
        You may need to adapt this if your installed SDK returns something different.
        """
        if resp is None:
            return ""
        # Try common shapes (best-effort)
        # 1) new-style: resp.output_text or resp.output or resp.candidates
        if hasattr(resp, "output_text"):
            return resp.output_text
        if isinstance(resp, dict):
            # common dict shapes
            # new chat responses: {'candidates': [{'content': [{'type':'output_text','text': '...'}]}], ...}
            if "candidates" in resp:
                cand = resp["candidates"]
                if isinstance(cand, list) and cand:
                    c0 = cand[0]
                    # candidate may have 'content' or 'text'
                    if isinstance(c0, dict):
                        if "content" in c0 and isinstance(c0["content"], list):
                            # find first output_text piece
                            for p in c0["content"]:
                                if isinstance(p, dict) and p.get("type") in ("output_text", "output"):
                                    txt = p.get("text") or p.get("payload") or ""
                                    if txt:
                                        return txt
                        if "text" in c0:
                            return c0["text"]
            # legacy: {'text': '...'}
            if "text" in resp and isinstance(resp["text"], str):
                return resp["text"]
            # new nested: {'output': [{'content': [{'text': '...'}]}]}
            if "output" in resp and isinstance(resp["output"], list) and resp["output"]:
                out0 = resp["output"][0]
                if isinstance(out0, dict) and "content" in out0 and isinstance(out0["content"], list):
                    for p in out0["content"]:
                        if isinstance(p, dict) and ("text" in p):
                            return p["text"]
        # 2) object with .output and .candidates attributes
        if hasattr(resp, "output") and isinstance(resp.output, list) and resp.output:
            first = resp.output[0]
            # try to find text inside
            if isinstance(first, dict) and "content" in first:
                for part in first["content"]:
                    if isinstance(part, dict) and "text" in part:
                        return part["text"]
        # 3) fallback to str()
        return str(resp)

    def call(self, prompt: str, temperature: t.Optional[float] = None, max_output_tokens: t.Optional[int] = None) -> str:
        """
        Generate text for the given prompt synchronously.
        Returns a plain string.
        """
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_output_tokens if max_output_tokens is None else max_output_tokens

        # Try multiple likely SDK call patterns (best-effort compatibility)
        # 1) genai.generate(...)
        try:
            resp = genai.generate(
                model=self.model,
                prompt=prompt,
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            text = self._extract_text_from_response(resp)
            if text:
                return text
        except Exception:
            # not all SDKs have genai.generate or it might throw for other reasons
            pass

        # 2) genai.chat.create(...) (chat-style API)
        try:
            resp = genai.chat.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            text = self._extract_text_from_response(resp)
            if text:
                return text
        except Exception:
            pass

        # 3) genai.text.generate(...) (another possible variant)
        try:
            resp = genai.text.generate(
                model=self.model,
                prompt=prompt,
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            text = self._extract_text_from_response(resp)
            if text:
                return text
        except Exception:
            pass

        # 4) As a final fallback, raise an error with guidance
        raise RuntimeError(
            "Gemini call failed. The installed `google.generativeai` SDK may have a different API shape. "
            "Check SDK docs for correct call pattern (generate / chat.create / text.generate)."
        )

########## End wrapper ##########

def setup_llm():
    """Set up the Gemini LLM wrapper for CrewAI with optimized settings."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    # Optimized model settings for faster responses
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    return GeminiLLM(
        api_key=google_api_key,
        model=model_name,
        temperature=0.3,  # Lower temperature for faster, more focused responses
        max_output_tokens=800  # Reduced tokens for faster generation
    )

# ========== DAILY PLANNER ==========
def setup_daily_planner_agents():
    """Set up the CrewAI agents for daily planning using Gemini via the wrapper."""
    llm = setup_llm()

    schedule_analyst = Agent(
        role='Daily Schedule Analyst',
        goal='Analyze daily tasks and create optimal time allocations',
        backstory="""You are an expert in time management and daily planning. 
        You excel at analyzing tasks, estimating time requirements, and creating 
        balanced daily schedules that maximize productivity while preventing burnout.""",
        verbose=False,  # Disabled verbosity for faster execution
        llm=llm,
        allow_delegation=False,  # Disabled delegation for faster responses
        max_iter=3  # Limit iterations for faster completion
    )

    productivity_coach = Agent(
        role='Productivity Coach',
        goal='Design productivity strategies and optimize daily routines',
        backstory="""You are a certified productivity coach with expertise in habit formation, 
        energy management, and work-life balance. You specialize in creating personalized 
        daily routines that help people achieve their goals efficiently.""",
        verbose=False,  # Disabled verbosity for faster execution
        llm=llm,
        max_iter=3  # Limit iterations for faster completion
    )

    routine_planner = Agent(
        role='Daily Routine Planner',
        goal='Create detailed daily schedules with time blocks and breaks',
        backstory="""You are a daily planning specialist who helps people organize their days 
        effectively. You create detailed schedules with optimal time blocks, break periods, 
        and flexibility for unexpected events.""",
        verbose=False,  # Disabled verbosity for faster execution
        llm=llm,
        max_iter=3  # Limit iterations for faster completion
    )

    return schedule_analyst, productivity_coach, routine_planner

def create_daily_plan(tasks_description: str):
    """Create a comprehensive daily plan using CrewAI agents."""
    schedule_analyst, productivity_coach, routine_planner = setup_daily_planner_agents()
    
    analysis_task = Task(
        description=f"""Analyze the following daily tasks and create time estimates:
{tasks_description}

Provide a structured analysis including:
1. Task priorities and urgency levels
2. Estimated time requirements for each task
3. Energy levels needed for different activities
4. Best times of day for each type of task
5. Potential conflicts and dependencies""",
        agent=schedule_analyst,
        expected_output="A detailed task analysis with time estimates and priority rankings."
    )

    strategy_task = Task(
        description="""Based on the task analysis, design productivity strategies and daily routines.
Consider:
- Optimal work sessions and break patterns
- Energy management techniques
- Focus and concentration strategies
- Time blocking methodologies
- Habit formation recommendations""",
        agent=productivity_coach,
        expected_output="Comprehensive productivity strategies tailored to the individual's needs and preferences.",
        context=[analysis_task]
    )

    schedule_task = Task(
        description="""Based on the analysis and strategies, create a detailed daily schedule.
Include:
- Hour-by-hour time blocks
- Break periods and meal times
- Buffer time for unexpected tasks
- Energy-appropriate task placement
- Flexibility for schedule adjustments""",
        agent=routine_planner,
        expected_output="A complete daily schedule with specific time blocks and built-in flexibility.",
        context=[strategy_task]
    )

    llm = setup_llm()
    crew = Crew(
        agents=[schedule_analyst, productivity_coach, routine_planner],
        tasks=[analysis_task, strategy_task, schedule_task],
        verbose=False,  # Disabled verbose output for better performance
        process=Process.sequential,
        manager_llm=llm,
        function_calling_llm=llm,
        max_rpm=100,  # Increase requests per minute limit
        cache=True  # Enable caching for faster responses
    )

    result = crew.kickoff()
    try:
        if isinstance(result, dict):
            return result.get("text") or str(result)
        return str(result)
    except Exception:
        return str(result)

# ========== REMINDER AGENT ==========
def setup_reminder_agents():
    """Set up the CrewAI agents for reminder management using Gemini via the wrapper."""
    llm = setup_llm()

    task_analyzer = Agent(
        role='Task Analysis Specialist',
        goal='Analyze tasks and extract key information for optimal reminder scheduling',
        backstory="""You are an expert in task management and productivity optimization. 
        You excel at breaking down complex tasks into actionable components and identifying 
        the optimal timing and frequency for reminders.""",
        verbose=False,  # Disabled verbosity for faster execution
        llm=llm,
        allow_delegation=False,  # Disabled delegation for faster responses
        max_iter=3  # Limit iterations for faster completion
    )

    scheduler = Agent(
        role='Intelligent Scheduler',
        goal='Create optimal reminder schedules based on task priorities and user preferences',
        backstory="""You are a scheduling expert who understands time management principles 
        and human productivity patterns. You specialize in creating reminder schedules that 
        maximize effectiveness while avoiding reminder fatigue.""",
        verbose=False,  # Disabled verbosity for faster execution
        llm=llm,
        max_iter=3  # Limit iterations for faster completion
    )

    reminder_strategist = Agent(
        role='Reminder Strategy Advisor',
        goal='Provide comprehensive reminder strategies and best practices',
        backstory="""You are a productivity coach who helps people develop effective reminder 
        systems. You provide actionable advice on reminder types, frequency, and delivery 
        methods to ensure tasks are completed on time.""",
        verbose=False,  # Disabled verbosity for faster execution
        llm=llm,
        max_iter=3  # Limit iterations for faster completion
    )

    return task_analyzer, scheduler, reminder_strategist

def create_reminder_plan(task_description: str, preferences: str = ""):
    """Create a comprehensive reminder plan using CrewAI agents."""
    task_analyzer, scheduler, reminder_strategist = setup_reminder_agents()
    
    analysis_task = Task(
        description=f"""Analyze the following task and extract key information for reminder scheduling:
Task: {task_description}
User Preferences: {preferences}

Provide a structured analysis including:
1. Task complexity and estimated duration
2. Key milestones and deadlines
3. Priority level and urgency
4. Dependencies and prerequisites
5. Potential obstacles or challenges""",
        agent=task_analyzer,
        expected_output="A detailed task analysis with scheduling considerations."
    )

    scheduling_task = Task(
        description="""Based on the task analysis, create an optimal reminder schedule.
Consider:
- Best times for reminders based on task type
- Frequency and spacing of reminders
- Escalation strategy for missed reminders
- Different reminder types (pre-task, progress check, deadline)
- Buffer time for unexpected delays""",
        agent=scheduler,
        expected_output="A comprehensive reminder schedule with timing and frequency recommendations.",
        context=[analysis_task]
    )

    strategy_task = Task(
        description="""Based on the analysis and schedule, provide reminder strategies and best practices.
Include:
- Reminder delivery methods (notifications, emails, etc.)
- Message content and tone recommendations
- Motivation and engagement techniques
- Tracking and completion verification
- Long-term habit formation advice""",
        agent=reminder_strategist,
        expected_output="Actionable reminder strategies and implementation recommendations.",
        context=[scheduling_task]
    )

    llm = setup_llm()
    crew = Crew(
        agents=[task_analyzer, scheduler, reminder_strategist],
        tasks=[analysis_task, scheduling_task, strategy_task],
        verbose=False,  # Disabled verbose output for better performance
        process=Process.sequential,
        manager_llm=llm,
        function_calling_llm=llm,
        max_rpm=100,  # Increase requests per minute limit
        cache=True  # Enable caching for faster responses
    )

    result = crew.kickoff()
    try:
        if isinstance(result, dict):
            return result.get("text") or str(result)
        return str(result)
    except Exception:
        return str(result)

# ========== MAIN APPLICATION ==========
def main():
    # Sidebar for API key input
    with st.sidebar:
        st.header("Configuration")
        st.info("Using Gemini API key from .env file")
        if st.button("Reload API Key"):
            load_dotenv(override=True)
            st.rerun()

        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This integrated application uses CrewAI to provide two specialized AI agents:
        - Daily Planner  
        - Reminder Agent
        
        All agents use Google's Gemini models for intelligent analysis and recommendations.
        """)

    if not os.getenv("GOOGLE_API_KEY"):
        st.error("🔑 GOOGLE_API_KEY not found. Please check your .env file and restart the application.")
        return

    # Agent Selection
    st.markdown('<div class="agent-selector">', unsafe_allow_html=True)
    
    agent_choice = st.selectbox(
        "Choose an Agent:",
        ["📅 Daily Planner", "⏰ Reminder Agent"],
        index=0,
        help="Select which specialized AI agent you want to use"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Display agent-specific information
    if agent_choice == "📅 Daily Planner":
        st.markdown('<div class="agent-card">', unsafe_allow_html=True)
        st.markdown("### 📅 Daily Planner")
        st.markdown("Create comprehensive daily plans and schedules with optimal time management and productivity strategies.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        with st.form("daily_planner_form"):
            tasks_description = st.text_area(
                "Describe your tasks and schedule for the day:",
                placeholder="E.g., I have team meeting at 10 AM, need to finish project report by 2 PM, workout session planned, grocery shopping, and dinner with family at 7 PM. I'm most productive in the morning and prefer to finish important tasks early.",
                height=150
            )
            submitted = st.form_submit_button("� Create Daily Plan")
            
            if submitted and tasks_description:
                with st.spinner("📅 Creating your daily plan. This may take a minute..."):
                    try:
                        result = create_daily_plan(tasks_description)
                        st.markdown("## Your Daily Plan")
                        result_str = str(result) if result else "No results returned"
                        st.markdown(
                            f"""
                            <div style="
                                max-height: 500px;
                                overflow-y: auto;
                                padding: 20px;
                                background-color: #ffffff;
                                border-radius: 8px;
                                border: 1px solid #e0e0e0;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                                line-height: 1.6;
                                color: #333333;
                                font-size: 14px;
                            ">
                                {result_str}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        st.success("Daily plan created successfully!")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        st.error("Please check your API key, model name and installed google.generativeai SDK version.")
            elif submitted and not tasks_description:
                st.warning("Please enter your tasks to create a daily plan.")

    else:  # Reminder Agent
        st.markdown('<div class="agent-card">', unsafe_allow_html=True)
        st.markdown("### ⏰ Reminder Agent")
        st.markdown("Create intelligent reminder systems and task management strategies with optimal scheduling and personalized approaches.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        with st.form("reminder_form"):
            task_description = st.text_area(
                "Describe the task you need reminders for:",
                placeholder="E.g., Complete the quarterly financial report by Friday, including data analysis, charts, and executive summary.",
                height=100
            )
            
            preferences = st.text_area(
                "Your preferences (optional):",
                placeholder="E.g., Prefer morning reminders, don't remind after 8 PM, like gentle reminders, need multiple reminders for important tasks",
                height=80
            )
            
            submitted = st.form_submit_button("⏰ Create Reminder Plan")
            
            if submitted and task_description:
                with st.spinner("⏰ Creating your intelligent reminder plan. This may take a minute..."):
                    try:
                        result = create_reminder_plan(task_description, preferences)
                        st.markdown("## Your Intelligent Reminder Plan")
                        result_str = str(result) if result else "No results returned"
                        st.markdown(
                            f"""
                            <div style="
                                max-height: 500px;
                                overflow-y: auto;
                                padding: 20px;
                                background-color: #ffffff;
                                border-radius: 8px;
                                border: 1px solid #e0e0e0;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                                line-height: 1.6;
                                color: #333333;
                                font-size: 14px;
                            ">
                                {result_str}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        st.success("Reminder plan created successfully!")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        st.error("Please check your API key, model name and installed google.generativeai SDK version.")
            elif submitted and not task_description:
                st.warning("Please enter a task description to create a reminder plan.")

if __name__ == "__main__":
    main()
