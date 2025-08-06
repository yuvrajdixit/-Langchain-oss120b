from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory

# 🔹 Load the OSS 120B Model from Hugging Face using pipeline
print("⏳ Loading GPT-OSS-120B model...")
pipe = pipeline(
    "text-generation",
    model="openai/gpt-oss-120b",
    torch_dtype="auto",
    device_map="auto",
    max_new_tokens=256,
)
print("✅ Model loaded successfully.")

# 🔹 Wrap the pipeline in LangChain's HuggingFacePipeline LLM
llm = HuggingFacePipeline(pipeline=pipe)

# 🔹 Example 1: Using LLMChain with a PromptTemplate
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} clearly and concisely.",
)

chain = LLMChain(llm=llm, prompt=prompt)

print("\n🔸 Running prompt chain:")
response = chain.run("quantum mechanics")
print("🧠 Response:\n", response)

# 🔹 Example 2: Using Conversational Memory
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

print("\n🔸 Running conversational memory chain:")
conversation.predict(input="Hello, who are you?")
final_response = conversation.predict(input="Explain black holes like I'm 5.")
print("🧠 Final Response:\n", final_response)
