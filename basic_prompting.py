
from langchain_core.prompts import PromptTemplate
from utils.ai_model import AiModel

model = AiModel()
llm = model.get_llm()

# Test the model
template = "Explain this concept simply and concisely: {concept}"
prompt_template = PromptTemplate.from_template(template=template)

#Let us create a chain that connects calls to different components
#user input is fed to the prompt template
#output of prompt template is fed to the llm and you get a response
llm_chain = prompt_template | llm
result = llm_chain.invoke({"concept" : "Prompting LLms"})
print(f"result=\n{result}")