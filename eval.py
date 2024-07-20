from datasets import Dataset
from dotenv import find_dotenv
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import answer_relevancy
from ragas.metrics import context_precision
from ragas.metrics import context_recall
from ragas.metrics import faithfulness

from data import DataLoader
from data import VectorStore
from model import QAModel

# Load environment variables from a .env file
load_dotenv(find_dotenv())

# Define a list of questions to query the model
questions = [
    "What is the property's address?",
    "Does this property accept pets?",
    "What is the cancellation policy?",
    "Can I vape at the property?",
    "Do you need a key to access the property?",
    "What is the total price for 1 guest staying two nights with a small pet?",
    "What time does the guest need to check out by?",
    "what is the name of the guest arriving on July 17?",
    "Is there street parking at the property?",
    "What's the home's WiFi password and where is the router located?",
    "Is there a coffeemaker provided and what is the contact's email?",
    "Is there a pet fee and what is the name of the guest's snail?",
    "What is the best place to get a bowl of chili near the rental?",
    "What should the guest do before they check-out?",
    "Where is the home's washing machine?",
    "Where are the fishing poles stored in the home?",
    "How many wine-glasses are provided?",
    "What star sign was the guest born under?",
    "Is the fireplace electric or gas-powered?",
]
ground_truths = [
    "The property's address is 402 4th St NE Washington, DC 20002. (sources: welcome_packet, listing)",
    "Yes, this property can accommodate a small dog or cat with an additional $50 pet fee. (sources: listing, conversation)",
    "Full refund if cancelled at least 15 days ahead of time. Otherwise no refund is provided. (sources: listing)",
    "We do not allow smoking of any kind (including vaping & e-cigarettes) in the home, anywhere on the property (including on balconies), or anywhere within 50 feet of the property; breaking this rule is grounds for termination of your stay and will incur a $500 cleaning fee. (sources: listing)",
    "No, you do not need a key. The home is equipped with an electronic smartlock, which can be accessed using the code 2048. Please note that the door automatically locks at 30 seconds. (sources: listing, welcome_packet)",
    "The total price for this reservation would be $650. The nightly price is $200, plus an additional $50 pet fee. (sources: listing)",
    "Check-out is any time before 11AM. (sources: welcome_packet, listing)",
    "The name of the guest arriving on July 17 is Patrick Star. (sources: conversation)",
    "There is limited street parking outside of the property. Guests should park in the green-signed zones to avoid tickets. (sources: welcome_packet)",
    "The WiFi password is Cortad0! and the router is located in the main dining room by the window. (sources: welcome_packet)",
    "The home is equipped with a Mr. Coffee drip coffee maker. The contact's email is bob@yahoo.mail. (sources: welcome_packet, listing)",
    "The pet fee is $50 and the snail's name is Gary. (sources: listing, conversation)",
    "The best place to get some chili is the famous Ben's Chili Bowl. (sources: welcome_packet)",
    "The guest should place all used sheets and towels in the washing machine and run it prior to their departure. They should do the same for all dirty dishes and cookware, placing them in the dishwasher and running it before leaving. (sources: welcome_packet, listing)",
    "The home's washing machine is located in the basement bathroom and is free for guests to use during their stay. (sources: welcome_packet)",
    "No information available. (sources: n/a)",
    "No information available. (sources: n/a)",
    "No information available. (sources: n/a)",
    "No information available. (sources: n/a)",
]


# Initialize lists to store answers and contexts
answers = []
contexts = []

# Load documents using DataLoader
data_loader = DataLoader()
documents = data_loader.load()

# Initialize the vector store and add documents to it
vector_store = VectorStore()
vector_store.add_documents(documents)

# Initialize the QA model with the vector store
qa = QAModel(vector_store=vector_store)

# Perform inference for each query in the questions list
for query in questions:
    answers.append(qa.answer(query))
    contexts.append(vector_store.get_context_texts(query))

# Prepare the data for evaluation
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths,
}

# Create a dataset from the data dictionary
dataset = Dataset.from_dict(data)

# Evaluate the model's performance using specified metrics
result = evaluate(
    dataset=dataset,
    is_async=False,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
)

# Convert the evaluation results to a pandas DataFrame and save to CSV and Excel
df = result.to_pandas()
df.to_csv("resources/results/eval.csv", index=False)
df.to_excel("resources/results/eval.xlsx", index=False)
