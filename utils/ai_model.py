import threading
from utils.app_config import AppConfig
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class AiModel:
    """
    Thread-safe Singleton class to hold Ai model.
    """

    _instance = None
    _lock = threading.Lock()  # Lock object to synchronize threads

    def __new__(cls):
        """
        Override __new__ to implement thread-safe singleton pattern.
        """
        if cls._instance is None:
            with cls._lock:  # Ensure only one thread can initialize the instance
                if cls._instance is None:  # Check again to ensure only one object is created
                    cls._instance = super(AiModel, cls).__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self):
        """
        Initialize the AiModel instance.
        This method is called only once during the lifetime of the singleton.
        """

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
        self.llm = HuggingFacePipeline(pipeline=pipe)

    def get_llm(self):
        return self.llm
