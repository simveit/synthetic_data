import openai
import instructor
from pydantic import BaseModel
import random
import yaml 

client = instructor.from_openai(openai.OpenAI())


def load_and_get_sample(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    topic = random.choice(data)
    example = random.choice(topic["example_types"])
    return {
        "topic": topic["topic"],
        "example_type": example["example_type"],
        "example_background": example["example_background"],
        "example_query": example["example_query"]
    }

profile = load_and_get_sample("user_profiles.yaml")

client_description = """# Millennia Bank Information

Millennia Bank, established in 2002 in Berlin, is a consumer finance institution serving customers across Europe. The bank offers various products such as checking and savings accounts, loans, and multiple credit card lines. One of its primary offerings is the **MillenniaCard**, which comes in three tiers: **Classic**, **Platinum**, and **Student**.

## MillenniaCard Details

- **Annual Fees**: Ranging from €30 to €90, depending on the tier.  
- **Interest Rates**: Starting at 14.99%, subject to creditworthiness and local regulations.  
- **Credit Limits**: Typically begin at €1,000 for Classic cards and can go up to €10,000 for Platinum; Student cards have lower limits.  
- **Rewards Program**: Points awarded for qualifying purchases, redeemable for statement credits, gift cards, or partner discounts.

Additional features include contactless payment options, an emergency replacement service, and a mobile app for account management.

## Tiers Overview

1. **Classic**  
   - Annual fee of €30  
   - Credit limits often range from €1,000 to €5,000  
   - Standard rewards rate for everyday spending

2. **Platinum**  
   - Annual fee of €90  
   - Credit limit can extend up to €10,000  
   - Enhanced perks (e.g., travel benefits, higher rewards)

3. **Student**  
   - Low or no annual fee  
   - Reduced credit limit, often up to €1,500  
   - Tailored for students aged 18–25

## Key Charges

- **Late Payment Fees**: Applied if the monthly balance is not settled by the due date.  
- **Foreign Transaction Fees**: Typically 2–3% for the Classic and Student tiers; Platinum users may receive lower or waived fees."""


class SyntheticQuestion(BaseModel):
    user_type: str
    user_background: str
    query: str

class Response(BaseModel):
    topic: str
    synthetic_question: SyntheticQuestion

synthetic_question = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": """We want to generate synthetic data to evaluate the capabilities of the chatbot of our client.
            Client description:
            `{{ client_description }}`
            To generate high quality data you will first come up with a user type and craft a small background about the user.
            You will then craft a query for that user. You will be given a topic that the query will be about.
            You will also be given an example for a user type, background and question for better understanding:
            Topic: {{ topic }}
            ```
            Example type: {{ example_type }}
            Example background: {{ example_background }}
            Example query: {{ example_query }}
            ```
            Please make sure that the question is on the topic. It is highly important the generate user type, background and query are about the topic.""",  
        },
    ],
    response_model=SyntheticQuestion,
    context={"client_description": client_description, "topic": profile["topic"],
             "example_type": profile["example_type"], "example_background": profile["example_background"],
             "example_query": profile["example_query"]},  
)

resp = Response(
    topic=profile["topic"],
    synthetic_question=synthetic_question
)

print(resp.topic)
print(profile["example_type"])
print(profile["example_background"])
print(profile["example_query"])
print(resp.synthetic_question.user_type)
print(resp.synthetic_question.user_background)
print(resp.synthetic_question.query)
# > Requesting an Additional Card
# > Roommate Sharing Bills
# > Iris splits rent and groceries with her roommate, hoping to consolidate spending on one card.
# > Could I get a second card for my roommate? We share expenses, and it’d be simpler to manage everything on one statement.
# > Student Cardholder
# > Mark is a university student who has just received his first MillenniaCard Student. He is eager to manage his finances better and enjoys using mobile apps for tracking expenses, especially since he has limited income from part-time work.
# > Can I get an additional card for my parents to help them manage shared expenses? It would make it easier for us to handle costs when they support me in my studies.
