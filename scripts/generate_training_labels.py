#!/usr/bin/env python3

import json
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import time

try:
    import google.generativeai as genai
except ImportError:
    os.system("pip install google-generativeai")
    import google.generativeai as genai
    
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash-exp')

MAX_API_CALLS = 17
EXAMPLES_PER_CALL = 30 

def sample_diverse_businesses(df, n=100):
    df['Risk_Category'] = pd.qcut(df['Risk_Score'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
    df['Growth_Category'] = pd.qcut(df['Revenue_Growth_Potential'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
    
    sample = df.groupby(['Industry', 'Growth_Stage'], group_keys=False).apply(
        lambda x: x.sample(min(len(x), max(1, n // 50)))
    ).sample(n=min(n, len(df)))
    
    return sample

def create_business_summary(row):
    challenges = ', '.join(eval(row['Main_Challenges'])) if isinstance(row['Main_Challenges'], str) else 'N/A'
    
    return {
        'business_id': row['Business_ID'],
        'industry': row['Industry'],
        'revenue': f"${row['Revenue_Last_12M']:,}",
        'location': row['Location'],
        'growth_stage': row['Growth_Stage'],
        'online_presence': row['Online_Presence'],
        'digital_payment': row['Digital_Payment_Adoption'],
        'young_customers_pct': f"{row['Pct_Young_Customers']:.1f}%",
        'market_spending_growth': f"{row['Consumer_Spending_Growth_YoY']}%",
        'competition_density': row['Competition_Density_Score'],
        'industry_trend': row['Industry_Growth_Trend'],
        'revenue_growth_potential': f"{row['Revenue_Growth_Potential']:.1f}",
        'risk_score': f"{row['Risk_Score']:.1f}",
        'digital_readiness': f"{row['Digital_Readiness']:.1f}",
        'regulatory_constraints': row['Regulatory_Constraints'][:150] + '...' if len(str(row['Regulatory_Constraints'])) > 150 else row['Regulatory_Constraints'],
        'customer_feedback': row['IG_Comments'][:150] + '...' if len(str(row['IG_Comments'])) > 150 else row['IG_Comments'],
        'main_challenges': challenges
    }

def generate_recommendations_batch(business_profiles, batch_num):
    
    prompt = f"""You are a business strategy consultant. Generate personalized, actionable recommendations for {len(business_profiles)} small businesses based on their profiles.

For EACH business below, provide:
1. 3-5 specific, actionable recommendations tailored to their situation
2. A confidence score (0-100) indicating how impactful these recommendations would be

Focus on:
- Digital transformation opportunities (based on online presence, customer demographics)
- Revenue growth strategies (based on market trends, growth potential)
- Risk mitigation (based on competition, regulatory environment)
- Operational efficiency improvements

Business Profiles:
{json.dumps(business_profiles, indent=2)}

OUTPUT FORMAT (must be valid JSON array):
[
  {{
    "business_id": "uuid-here",
    "recommendations": [
      "Specific recommendation 1...",
      "Specific recommendation 2...",
      "Specific recommendation 3..."
    ],
    "confidence_score": 85
  }},
  ...
]

Generate exactly {len(business_profiles)} recommendation objects. Be specific and actionable."""

    try:
        print(f"Batch {batch_num}/{MAX_API_CALLS}: generating recommendations for {len(business_profiles)} businesses")
        response = model.generate_content(prompt)
        
        response_text = response.text.strip()

        if response_text.startswith('```json'):
            response_text = response_text[7:]
        elif response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        response_text = response_text.strip().replace('\n\n', '\n')
        
        recommendations = json.loads(response_text)
        
        if isinstance(recommendations, list):
            print(f"Batch {batch_num}/{MAX_API_CALLS}: received {len(recommendations)} recommendations")
            return recommendations
        else:
            print("Unexpected response format")
            return []
            
    except Exception as e:
        print(f"Error: {e}")
        return []

def main():
    print("Generating labeled training data for fine-tuning...")
    print(f"Target: {MAX_API_CALLS * EXAMPLES_PER_CALL} examples in {MAX_API_CALLS} batches of {EXAMPLES_PER_CALL}\n")
    
    df = pd.read_csv('/Users/siddu/Developer/Business-Insights-Recommender/data/processed/engineered_business_data.csv')
    all_training_examples = []
    
    for i in range(MAX_API_CALLS):
        batch_df = sample_diverse_businesses(df, n=EXAMPLES_PER_CALL)
        
        business_profiles = [create_business_summary(row) for _, row in batch_df.iterrows()]
        
        batch_recommendations = generate_recommendations_batch(business_profiles, i + 1)
        
        if batch_recommendations:
            all_training_examples.extend(batch_recommendations)
        
        if i < MAX_API_CALLS - 1:
            time.sleep(2)
    
    print(f"\n{'='*60}")
    print("Training data generation complete")
    print(f"Total labeled examples: {len(all_training_examples)}")
    
    output_file = '/Users/siddu/Developer/Business-Insights-Recommender/data/processed/training_labels.jsonl'
    with open(output_file, 'w') as f:
        for example in all_training_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Saved to: {output_file}")
        
    rec_counts = [len(ex['recommendations']) for ex in all_training_examples]
    conf_scores = [ex['confidence_score'] for ex in all_training_examples]
    
    print("\nSummary:")
    print(f"Avg recommendations per business: {np.mean(rec_counts):.1f}")
    print(f"Avg confidence score: {np.mean(conf_scores):.1f}")
    print(f"Min/Max confidence: {min(conf_scores)}/{max(conf_scores)}")

if __name__ == "__main__":
    main()
