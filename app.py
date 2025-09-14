import streamlit as st
import pandas as pd
import random
from datetime import datetime
from openai import OpenAI
import os, json

# --- Configure OpenAI ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to parse user request via OpenAI (fallback if unavailable)
def parse_request_with_openai(prompt: str):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",   # or "gpt-4.1"
            messages=[
                {"role": "system", "content": "You are an assistant that extracts structured filters for employee recommendation."},
                {"role": "user", "content": f"Parse this project request into JSON filters:\n{prompt}"}
            ],
            temperature=0,
            max_tokens=300
        )
        content = response.choices[0].message.content.strip()
        return json.loads(content)
    except Exception as e:
        return None

# --- Local fallback parser ---
def parse_request_local(prompt: str):
    parsed = {
        "role": [],
        "skills": [],
        "location": None,
        "budget": None,
        "availability_before": None,
        "past_projects": [],
        "top_n": None
    }

    text = prompt.lower()

    if "devops" in text:
        parsed["role"].append("DevOps Engineer")
    if "iot" in text:
        parsed["past_projects"].append("IoT")
    if "calgary" in text:
        parsed["location"] = "Calgary"
    if "top 5" in text:
        parsed["top_n"] = 5
    elif "top 10" in text:
        parsed["top_n"] = 10

    return parsed


# --- Scoring logic ---
# --- Scoring logic ---
def score_employees(df, filters):
    results = []
    for _, row in df.iterrows():
        score = 0
        reasons = []

        # Role
        if filters.get("role") and row["role"] in filters["role"]:
            score += 20
            reasons.append(f"Role '{row['role']}' fits project requirement")

        # Skills
        if filters.get("skills"):
            matched_skills = [s for s in filters["skills"] if s.lower() in row["skills"].lower()]
            if matched_skills:
                score += 15 * len(matched_skills)
                reasons.append(f"Employee has required skills: {', '.join(matched_skills)}")

        # Location
        if filters.get("location") and filters["location"].lower() in row["location"].lower():
            score += 10
            reasons.append(f"Located in/near {filters['location']}")

        # Past projects
        if filters.get("past_projects"):
            matched_projects = [p for p in filters["past_projects"] if p.lower() in row["past_projects"].lower()]
            if matched_projects:
                score += 10 * len(matched_projects)
                reasons.append(f"Experience in similar projects: {', '.join(matched_projects)}")

        # Performance & utilization
        score += row["performance_score"] * 2
        score += row["billable_utilization_pct"] / 10
        reasons.append(f"High performance score ({row['performance_score']}) and utilization {row['billable_utilization_pct']}%")

        # Availability
        if filters.get("availability_before"):
            try:
                project_date = pd.to_datetime(filters["availability_before"])
                available_date = pd.to_datetime(row["availability_start_date"])
                if available_date <= project_date:
                    score += 10
                    reasons.append(f"Available before required date {project_date.date()}")
            except Exception:
                pass

        results.append({
            "employee_id": row["employee_id"],
            "name": row["name"],
            "role": row["role"],
            "annual_cost": row["annual_cost"],
            "location": row["location"],
            "performance_score": row["performance_score"],
            "experience_years": row["experience_years"],
            "availability_start_date": row["availability_start_date"],
            "stake_tier": row["stake_tier"],
            "score": score,
            "reasons": " | ".join(reasons) if reasons else "General fit based on performance and utilization"
        })

    return pd.DataFrame(results).sort_values(by="score", ascending=False)


# --- Streamlit App ---
st.title("HR Talent Lens - AI Recommendation Engine")

uploaded_file = st.file_uploader("Upload Employee CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Describe your project")
    user_prompt = st.text_area("Enter project goal, budget, location, skills, etc.")

    # Fallback to slider if top_n not in prompt
    top_n_default = st.slider("Number of employees to recommend", 5, 50, 10)

    if st.button("Recommend"):
        filters = parse_request_with_openai(user_prompt) or parse_request_local(user_prompt)

        # Use slider if prompt did not specify top_n
        top_n = filters.get("top_n") if filters.get("top_n") else top_n_default

        scored_df = score_employees(df, filters).reset_index(drop=True)
        scored_df.index = scored_df.index + 1  # sequential numbering

        st.write(f"### Top {top_n} Recommendations")
        st.dataframe(scored_df.head(top_n)[[
            "score", "reasons", "employee_id", "name", "role",
            "annual_cost", "location", "performance_score",
            "experience_years", "availability_start_date", "stake_tier"
        ]])

        csv = scored_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results", csv, "recommendations.csv", "text/csv")

else:
    st.info("Please upload an employee CSV file to begin.")
