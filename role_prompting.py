from langchain_core.prompts import ChatPromptTemplate
from utils.ai_model import AiModel

model = AiModel()
llm = model.get_llm()

#system role defines how the model should behave
#human role is used for providing user inputs
#ai role is used for defining the model response
template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a calculator that responds with math."),
        ("human", "Answer this math question: What is two plus two?"),
        ("ai","2+2=4"),
        ("human", "Answer this math question: {math}")
    ]
)

llm_chain = template | llm
result = llm_chain.invoke({"math" : "What is five times five"})
print(f"result=\n{result}")
