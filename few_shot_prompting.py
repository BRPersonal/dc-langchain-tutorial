from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from utils.ai_model import AiModel

model = AiModel()
llm = model.get_llm()

example_prompt = PromptTemplate.from_template("Question: {question}\n{answer}")
examples = [
    {
        "question": "What is the capital of Italy?",
        "answer": "Rome"
    },
    {
        "question": "What is the capital of India?",
        "answer": "New Delhi"
    }

]
prompt_template = FewShotPromptTemplate(
    examples = examples,
    example_prompt = example_prompt,
    suffix="Question: {input}",
    input_variables=["input"]
)

llm_chain = prompt_template | llm
result = llm_chain.invoke({"input": "What is the name of Henry Cambell's dog?"})
print(f"result=\n{result}")
