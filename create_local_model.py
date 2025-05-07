from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from utils.AppConfig import AppConfig

#load AppConfig
_ = AppConfig()

# Load tokenizer and model
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  #"meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

# Create text-generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1000,
    do_sample=True,
    temperature=0.7,
)

# Create LangChain wrapper
llm = HuggingFacePipeline(pipeline=pipe)

# Test the model
template = "Explain this concept simply and concisely: {concept}"
prompt_template = PromptTemplate.from_template(template=template)

#Let us create a chain that connects calls to different components
llm_chain = prompt_template | llm
result = llm_chain.invoke({"concept" : "Prompting LLms"})
print(f"result=\n{result}")