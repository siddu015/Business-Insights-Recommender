#!/usr/bin/env python3

import json
import uuid
import random
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import time
import os
from dotenv import load_dotenv

try:
    import google.generativeai as genai
except ImportError:
    os.system("pip install google-generativeai")
    import google.generativeai as genai
    
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
TARGET_BUSINESS_RECORDS = 50000

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash-exp')  

INDUSTRIES = [
    "Retail", "Food Services", "Healthcare", "Education", "Tech", 
    "Construction", "Beauty/Wellness", "Professional Services", 
    "Manufacturing", "Agriculture", "Transportation", "Real Estate", 
    "Entertainment", "Non-profit", "Other"
]

LOCATIONS = [
    "New York, NY, USA", "Los Angeles, CA, USA", "Chicago, IL, USA", "Houston, TX, USA",
    "London, England, UK", "Toronto, ON, Canada", "Sydney, NSW, Australia",
    "Mumbai, Maharashtra, India", "Berlin, Germany", "Paris, France", "Tokyo, Japan",
    "Sao Paulo, Brazil", "Mexico City, Mexico", "Lagos, Nigeria", "Cairo, Egypt"
]

ONLINE_PRESENCE = ["Low", "Medium", "High"]
DIGITAL_PAYMENT = ["Cash-only", "Basic", "Advanced"]
GROWTH_STAGES = ["Startup", "Growing", "Mature", "Declining"]

class FastSyntheticDataGenerator:
    def __init__(self):
        self.text_templates = {}
        
    def generate_bulk_templates(self):
        """Generate text templates in bulk with fewer API calls"""
        print("Generating text templates (300 per field)...")
        
        regulatory_prompt = """Generate exactly 50 diverse regulatory constraint descriptions for small businesses. Each should be 2-3 sentences. Cover different industries (retail, healthcare, food, tech, etc.) and locations (US, EU, Asia, etc.). Output as a JSON array of strings.

Example format: ["Description 1...", "Description 2...", ...]"""

        ig_comments_prompt = """Generate exactly 50 diverse Instagram comment summaries for small businesses. Each should be 2-3 sentences reflecting customer feedback. Cover different industries and customer demographics. Output as a JSON array of strings.

Example format: ["Comments about great service...", "Mixed reviews on pricing...", ...]"""

        challenges_prompt = """Generate exactly 50 diverse business challenge sets. Each should be a JSON array of exactly 3 specific challenges. Cover different growth stages and industries. Output as a JSON array of arrays.

Example format: [["Challenge 1", "Challenge 2", "Challenge 3"], ["Challenge A", "Challenge B", "Challenge C"], ...]"""

        try:
            self.text_templates['regulatory'] = []
            self.text_templates['ig_comments'] = []
            self.text_templates['challenges'] = []
            
            print("Generating regulatory templates...")
            for i in range(6):
                reg_response = model.generate_content(regulatory_prompt)
                templates = self.parse_json_array(reg_response.text)
                self.text_templates['regulatory'].extend(templates)
                print(f"  Batch {i+1}/6: {len(templates)} templates")
                time.sleep(1)
            
            print(f"Total regulatory templates: {len(self.text_templates['regulatory'])}")
            
            print("Generating IG comment templates...")
            for i in range(6):
                ig_response = model.generate_content(ig_comments_prompt)
                templates = self.parse_json_array(ig_response.text)
                self.text_templates['ig_comments'].extend(templates)
                print(f"  Batch {i+1}/6: {len(templates)} templates")
                time.sleep(1)
            
            print(f"Total IG comment templates: {len(self.text_templates['ig_comments'])}")
                
            print("Generating challenge templates...")
            for i in range(6):
                challenges_response = model.generate_content(challenges_prompt)
                templates = self.parse_json_array(challenges_response.text)
                self.text_templates['challenges'].extend(templates)
                print(f"  Batch {i+1}/6: {len(templates)} templates")
                time.sleep(1)
            
            print(f"Total challenge templates: {len(self.text_templates['challenges'])}")
            
        except Exception as e:
            print(f"Error generating templates: {e}")
            self.create_fallback_templates()

    def parse_json_array(self, response_text: str) -> List:
        """Parse JSON array response"""
        try:
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            data = json.loads(response_text)
            return data if isinstance(data, list) else []
        except:
            return []

    def create_fallback_templates(self):
        """Create fallback templates if API fails"""
        self.text_templates = {
            'regulatory': [
                "Must comply with local business licensing and health department regulations.",
                "Subject to industry-specific safety standards and regular inspections.",
                "Required to maintain proper insurance coverage and tax compliance.",
                "Must follow employment laws and workplace safety regulations.",
                "Subject to environmental compliance and waste disposal requirements."
            ] * 10,  
            
            'ig_comments': [
                "Customers praise excellent service but mention higher prices than competitors.",
                "Mixed reviews on product quality, with most appreciating friendly staff.",
                "Great location and atmosphere, some complaints about wait times.",
                "Loyal customer base loves personalized service and community involvement.",
                "Recent improvements in online ordering have received positive feedback."
            ] * 10,
            
            'challenges': [
                ["Rising operational costs", "Attracting younger customers", "Competition from online retailers"],
                ["Staff retention issues", "Supply chain disruptions", "Digital marketing effectiveness"],
                ["Cash flow management", "Regulatory compliance burden", "Technology infrastructure needs"],
                ["Customer acquisition costs", "Inventory management", "Seasonal demand fluctuations"],
                ["Brand differentiation", "Scaling operations efficiently", "Market saturation concerns"]
            ] * 10
        }

    def generate_market_data_bulk(self):
        """Generate market data for ALL industry-location combinations using API"""
        print("Generating comprehensive market data for all Industry × Location combinations via API...")
        print(f"Total combinations to generate: {len(LOCATIONS) * len(INDUSTRIES)} (15 locations × 15 industries)")
        
        market_data = []
        batch_size = 15  # Generate 15 at a time (1 location × all industries)
        
        for location_idx, location in enumerate(LOCATIONS):
            print(f"Generating market data for {location} ({location_idx + 1}/{len(LOCATIONS)})...")
            
            market_prompt = f"""Generate realistic market data for {location} across ALL of these industries: {', '.join(INDUSTRIES)}.

For EACH industry, provide realistic market conditions based on actual economic trends for that location. Output as a JSON array with exactly {len(INDUSTRIES)} objects, one per industry, with these exact fields:

- Location: "{location}"
- Industry: One of {INDUSTRIES}
- Consumer_Spending_Growth_YoY: Realistic number between -5 and 12 based on actual market trends
- Competition_Density_Score: Integer 1-10 based on actual market saturation in {location}
- Regulatory_Changes_Upcoming: 2-3 sentence description of real/realistic upcoming regulations
- Digital_Adoption_Rate: Number 0-100 based on actual digital trends in {location}
- Local_Economic_Health_Score: Integer 1-10 based on actual economic health of {location}
- Industry_Growth_Trend: "Growing", "Stable", or "Declining" based on real industry trends

Output exactly {len(INDUSTRIES)} market data objects as a JSON array. Be realistic based on {location}'s actual economic conditions."""

            try:
                response = model.generate_content(market_prompt)
                batch_data = self.parse_json_array(response.text)
                
                if batch_data and len(batch_data) > 0:
                    market_data.extend(batch_data)
                    print(f"  ✓ Generated {len(batch_data)} records via API")
                else:
                    # Fallback for this location
                    fallback = self._create_fallback_for_location(location)
                    market_data.extend(fallback)
                    print(f"  ⚠ Used fallback for {location}")
                    
            except Exception as e:
                print(f"  ⚠ API error for {location}: {e}, using fallback")
                fallback = self._create_fallback_for_location(location)
                market_data.extend(fallback)
                    
            time.sleep(0.5)
        
        print(f"\nGenerated {len(market_data)} total market data records")
        return market_data
    
    def _create_fallback_for_location(self, location):
        """Create fallback market data for one location across all industries"""
        fallback_data = []
        city = location.split(',')[0]
        
        for industry in INDUSTRIES:
            fallback_data.append({
                "Location": location,
                "Industry": industry,
                "Consumer_Spending_Growth_YoY": round(random.uniform(-5, 12), 1),
                "Competition_Density_Score": random.randint(1, 10),
                "Regulatory_Changes_Upcoming": self._generate_regulatory_text(location, industry),
                "Digital_Adoption_Rate": round(random.uniform(45, 98), 1),
                "Local_Economic_Health_Score": random.randint(3, 10),
                "Industry_Growth_Trend": random.choice(["Growing", "Stable", "Declining"])
            })
        return fallback_data
    
    def _generate_regulatory_text(self, location, industry):
        """Generate regulatory text for a location-industry combo"""
        city = location.split(',')[0]
        
        reg_templates = [
            f"New {industry.lower()} regulations in {city} focusing on consumer protection and compliance standards. Digital reporting requirements are being updated.",
            f"{city} is implementing stricter {industry.lower()} licensing requirements. Environmental and labor law changes are anticipated.",
            f"Upcoming {industry.lower()} reforms in {city} include data privacy rules and operational transparency mandates.",
            f"Regulatory framework for {industry.lower()} in {city} is evolving with emphasis on sustainability and ethical practices.",
            f"{city} authorities are updating {industry.lower()} compliance standards. New tax incentives may also be introduced.",
        ]
        
        return random.choice(reg_templates)

    def create_fallback_market_data(self):
        """Create fallback market data"""
        market_data = []
        for location in LOCATIONS[:15]:
            for industry in random.sample(INDUSTRIES, 2):  
                market_data.append({
                    "Location": location,
                    "Industry": industry,
                    "Consumer_Spending_Growth_YoY": round(random.uniform(-5, 10), 1),
                    "Competition_Density_Score": random.randint(3, 8),
                    "Regulatory_Changes_Upcoming": f"Upcoming {industry.lower()} regulations in {location.split(',')[0]} focusing on digital compliance and consumer protection.",
                    "Digital_Adoption_Rate": round(random.uniform(40, 85), 1),
                    "Local_Economic_Health_Score": random.randint(5, 9),
                    "Industry_Growth_Trend": random.choice(["Growing", "Stable", "Declining"])
                })
        return market_data

    def generate_business_data_fast(self):
        """Generate 50K business records efficiently using templates"""
        print("Generating 50K business records...")
        
        businesses = []
        
        revenue_ranges = {
            "Tech": (80000, 2000000),
            "Healthcare": (100000, 1500000),
            "Retail": (50000, 800000),
            "Food Services": (60000, 500000),
            "Professional Services": (70000, 1200000),
            "Construction": (90000, 1800000),
            "Manufacturing": (120000, 3000000),
            "Other": (50000, 1000000)
        }
        
        for i in range(TARGET_BUSINESS_RECORDS):
            if i % 10000 == 0:
                print(f"Generated {i} records...")
            
            industry = random.choices(INDUSTRIES, weights=[0.15, 0.12, 0.10, 0.08, 0.12, 0.08, 0.06, 0.09, 0.05, 0.03, 0.04, 0.04, 0.02, 0.01, 0.01])[0]
            location = random.choice(LOCATIONS)
            
            rev_range = revenue_ranges.get(industry, (50000, 1000000))
            revenue = int(np.random.lognormal(np.log(rev_range[0] * 2), 0.7))
            revenue = max(rev_range[0], min(rev_range[1], revenue))
            
            if industry in ["Tech", "Entertainment"]:
                age_dist = {"18-25": round(random.uniform(25, 40), 1), "26-40": round(random.uniform(35, 50), 1), 
                           "41-55": round(random.uniform(15, 25), 1), "55+": round(random.uniform(5, 15), 1)}
            elif industry in ["Healthcare", "Professional Services"]:
                age_dist = {"18-25": round(random.uniform(10, 20), 1), "26-40": round(random.uniform(30, 45), 1), 
                           "41-55": round(random.uniform(30, 40), 1), "55+": round(random.uniform(15, 30), 1)}
            else:
                age_dist = {"18-25": round(random.uniform(15, 30), 1), "26-40": round(random.uniform(35, 50), 1), 
                           "41-55": round(random.uniform(25, 35), 1), "55+": round(random.uniform(10, 25), 1)}
            
            total = sum(age_dist.values())
            for key in age_dist:
                age_dist[key] = round(age_dist[key] / total * 100, 1)
            
            young_customers = age_dist["18-25"] + age_dist["26-40"]
            if young_customers > 65 and industry in ["Tech", "Retail", "Entertainment"]:
                online_presence = random.choices(["Low", "Medium", "High"], weights=[0.1, 0.3, 0.6])[0]
                digital_payment = random.choices(["Cash-only", "Basic", "Advanced"], weights=[0.05, 0.25, 0.7])[0]
            elif young_customers < 40 or industry in ["Agriculture", "Construction"]:
                online_presence = random.choices(["Low", "Medium", "High"], weights=[0.6, 0.3, 0.1])[0]
                digital_payment = random.choices(["Cash-only", "Basic", "Advanced"], weights=[0.4, 0.5, 0.1])[0]
            else:
                online_presence = random.choices(["Low", "Medium", "High"], weights=[0.3, 0.5, 0.2])[0]
                digital_payment = random.choices(["Cash-only", "Basic", "Advanced"], weights=[0.2, 0.6, 0.2])[0]
            
            if revenue < 100000:
                growth_stage = random.choices(["Startup", "Growing", "Mature", "Declining"], weights=[0.5, 0.3, 0.15, 0.05])[0]
            elif revenue > 800000:
                growth_stage = random.choices(["Startup", "Growing", "Mature", "Declining"], weights=[0.05, 0.25, 0.6, 0.1])[0]
            else:
                growth_stage = random.choices(["Startup", "Growing", "Mature", "Declining"], weights=[0.2, 0.4, 0.3, 0.1])[0]
            
            business = {
                "Business_ID": str(uuid.uuid4()),
                "Industry": industry,
                "Revenue_Last_12M": revenue,
                "Location": location,
                "Customer_Age_Distribution": age_dist,
                "Regulatory_Constraints": random.choice(self.text_templates.get('regulatory', ["Standard business compliance required."])),
                "IG_Comments": random.choice(self.text_templates.get('ig_comments', ["Positive customer feedback overall."])),
                "Online_Presence": online_presence,
                "Digital_Payment_Adoption": digital_payment,
                "Main_Challenges": random.choice(self.text_templates.get('challenges', [["Market competition", "Cost management", "Customer acquisition"]])),
                "Growth_Stage": growth_stage
            }
            
            businesses.append(business)
        
        return businesses

    def save_templates(self):
        """Save generated templates to a separate file for future reference"""
        print("Saving templates...")
        
        templates_data = {
            "generation_timestamp": pd.Timestamp.now().isoformat(),
            "template_counts": {
                "regulatory": len(self.text_templates.get('regulatory', [])),
                "ig_comments": len(self.text_templates.get('ig_comments', [])),
                "challenges": len(self.text_templates.get('challenges', []))
            },
            "templates": self.text_templates
        }
        
        with open('/Users/siddu/Developer/Business-Insights-Recommender/data/processed/generated_templates.json', 'w') as f:

            
            json.dump(templates_data, f, indent=2)
        
        print(f"Saved templates to data/processed/generated_templates.json")
        print(f"  - Regulatory: {templates_data['template_counts']['regulatory']} templates")
        print(f"  - IG Comments: {templates_data['template_counts']['ig_comments']} templates")
        print(f"  - Challenges: {templates_data['template_counts']['challenges']} templates")

    def save_datasets(self, businesses: List[Dict], market_data: List[Dict]):
        """Save datasets to CSV files"""
        print("Saving datasets...")
        
        business_df = pd.DataFrame(businesses)
        business_df.to_csv('/Users/siddu/Developer/Business-Insights-Recommender/data/raw/business_data.csv', index=False)
        print(f"Saved {len(business_df)} business records to data/raw/business_data.csv")
        
        market_df = pd.DataFrame(market_data)
        market_df.to_csv('/Users/siddu/Developer/Business-Insights-Recommender/data/raw/market_data.csv', index=False)
        print(f"Saved {len(market_df)} market records to data/raw/market_data.csv")
        
        print("\nSample Business Data:")
        print(business_df.head(3).to_string())
        print(f"\nBusiness Data Shape: {business_df.shape}")
        print(f"Market Data Shape: {market_df.shape}")

def main():
    """Main execution function"""
    print("Starting Fast Business Insights Recommender - Synthetic Data Generation")
    print("=" * 70)
    
    start_time = time.time()
    generator = FastSyntheticDataGenerator()
    
    try:
        
        generator.generate_bulk_templates()
        
        
        generator.save_templates()
        
        
        market_data = generator.generate_market_data_bulk()
        
        
        businesses = generator.generate_business_data_fast()
        
        generator.save_datasets(businesses, market_data)
        
        end_time = time.time()
        print(f"\n{'='*70}")
        print(f"Total generation time: {end_time - start_time:.2f} seconds")
        print(f"Records per second: {TARGET_BUSINESS_RECORDS / (end_time - start_time):.0f}")
        print(f"\nFiles created:")
        print(f"  ✓ business_data.csv (50,000 records)")
        print(f"  ✓ market_data.csv (30 records)")
        print(f"  ✓ generated_templates.json (~300 templates per field)")
        
    except Exception as e:
        print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()
