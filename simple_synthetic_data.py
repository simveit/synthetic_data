import openai
import instructor
from pydantic import BaseModel

client = instructor.from_openai(openai.OpenAI())


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
    chain_of_thought: str
    query: str

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": """Write a potential user query to a chatbot for the following product:
            `{{ data }}`""",  
        },
    ],
    response_model=SyntheticQuestion,
    context={"data": client_description},  
)

print(resp.chain_of_thought)
print(resp.query)

# > The user is likely interested in understanding the features, benefits, or specifics of the MillenniaCard. They could be considering which tier of the MillenniaCard is best suited for their needs, or they may have questions about fees, interest rates, or rewards. A suitable query would reflect an inquiry about the different card options and their associated benefits to help in making a decision.
# > What are the differences between the Classic, Platinum, and Student tiers of the MillenniaCard, and what benefits does each one offer?