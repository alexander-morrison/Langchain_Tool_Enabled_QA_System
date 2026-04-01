import os                                               
import dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import FewShotPromptTemplate

# load the api
dotenv.load_dotenv()

# create and store the model
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define example planet descriptions for few-shot learning
examples = [
    {
        "planet": "Jupiter",
        "answer": """Jupiter is the largest planet in the solar system.
Physical Characteristics: It is a gas giant composed mainly of hydrogen and helium.
Notable Features: The Great Red Spot and over 79 known moons, including Ganymede.
Scientific Significance: Its massive gravity helps protect inner planets from asteroids.
Fun Fact: A day on Jupiter lasts only about 10 hours."""
    },
    {
        "planet": "Mars",
        "answer": """Mars is the fourth planet from the Sun.
Physical Characteristics: A rocky planet with a thin carbon dioxide atmosphere.
Notable Features: Olympus Mons (largest volcano) and Valles Marineris canyon system.
Scientific Significance: A major focus of exploration due to the search for past life.
Fun Fact: Mars has the largest dust storms in the solar system."""
    }
]

# Define formatting structure for each example
example_prompt = PromptTemplate.from_template(
    "Planet: {planet}\nDescription:\n{answer}\n"
)

# Create few-shot prompt template with structured instructions.
few_shot_template = FewShotPromptTemplate(
    # give the model the Jupiter and Mars examples.
    examples=examples,
    # Tells LangChain how to format each example.
    example_prompt=example_prompt,
    # Instructions that guide the response structure.
    prefix=(
        "Provide detailed information about the given planet.\n"
        "Include the following sections:\n"
        "- Physical Characteristics\n"
        "- Notable Features\n"
        "- Scientific Significance\n"
        "- Fun Fact\n\n"
    ),
    # This is where the new planet goes.
    suffix="Planet: {planet}\nDescription:\n", 
    # Declares what variable the template expects.
    input_variables=["planet"]
)

# Get planet name from user input
planet_name = input()

# Generate formatted few-shot prompt
prompt = few_shot_template.invoke({"planet": planet_name})

# generate response from model
response = llm.invoke(prompt)

# print generated answer
print(response.content)
