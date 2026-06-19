from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
# TOOL 1: Attendance Calculator
@tool
def attendance_calculator(total_classes: int, attended_classes: int) -> str:
    """Calculate attendance percentage and exam eligibility."""

    percentage = (attended_classes / total_classes) * 100

    if percentage >= 75:
        status = "Eligible for Exam"
    else:
        status = "Not Eligible for Exam"

    return (
        f"Attendance Percentage: {percentage:.2f}%\n"
        f"Status: {status}"
    )
# TOOL 2: Result Calculator
@tool
def result_calculator(
    sub1: int,
    sub2: int,
    sub3: int,
    sub4: int,
    sub5: int
) -> str:
    """Calculate average marks, grade and pass/fail status."""
    average = (sub1 + sub2 + sub3 + sub4 + sub5) / 5
    if average >= 90:
        grade = "A"
    elif average >= 75:
        grade = "B"
    elif average >= 60:
        grade = "C"
    else:
        grade = "D"
    result = "Pass" if average >= 50 else "Fail"
    return (
        f"Average Marks: {average:.2f}\n"
        f"Grade: {grade}\n"
        f"Result: {result}"
    )
# TOOL 3: Fee Balance Calculator
@tool
def fee_balance_calculator(
    total_fee: float,
    amount_paid: float
) -> str:
    """Calculate pending fee amount."""

    pending = total_fee - amount_paid

    return f"Pending Fee: ₹{pending:.2f}"
# TOOL 4: Library Fine Calculator
@tool
def library_fine_calculator(delayed_days: int) -> str:
    """Calculate library fine amount."""

    fine = delayed_days * 5

    return f"Fine Amount: ₹{fine}"
# TOOL 5: Hostel Fee Calculator
@tool
def hostel_fee_calculator(
    monthly_fee: float,
    months_stayed: int
) -> str:
    """Calculate hostel fee."""

    total = monthly_fee * months_stayed

    return f"Total Hostel Fee: ₹{total:.2f}"
# BONUS TOOL : Student Information
students = {
    "101": {
        "name": "Rahul Sharma",
        "branch": "CSE",
        "year": "3rd Year"
    },
    "102": {
        "name": "Priya Singh",
        "branch": "ECE",
        "year": "2nd Year"
    },
    "103": {
        "name": "Amit Kumar",
        "branch": "Mechanical",
        "year": "4th Year"
    }
}
@tool
def student_information(student_id: str) -> str:
    """Retrieve student information using student ID."""

    student = students.get(student_id)

    if student:
        return (
            f"Name: {student['name']}\n"
            f"Branch: {student['branch']}\n"
            f"Year: {student['year']}"
        )

    return "Student Not Found"
# LLM
llm = ChatOllama(
    model="llama3.1",
    temperature=0
)
# TOOLS LIST
tools = [
    attendance_calculator,
    result_calculator,
    fee_balance_calculator,
    library_fine_calculator,
    hostel_fee_calculator,
    student_information
]
# PROMPT
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an AI College Assistant.
Use the provided tools whenever calculations are required.
Always return the exact tool result.
Do not perform calculations yourself.
"""
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ]
)
# AGENT
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)
# MAIN LOOP
print("=" * 60)
print("AI COLLEGE ASSISTANT USING LANGCHAIN TOOL CALLING AGENT")
print("=" * 60)
print("\nType 'exit' to quit.\n")
while True:
    query = input("Ask Your Question: ")
    if query.lower() == "exit":
        print("Goodbye!")
        break
    try:
        response = agent_executor.invoke(
            {"input": query}
        )
        print("\nResponse:")
        print(response["output"])
    except Exception as e:
        print(f"\nError: {e}")