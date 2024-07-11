import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load student profiles from CSV
df_profiles = pd.read_csv("student_data.csv")

# Ensure that the profiles in the dataset have skills as strings
df_profiles['Skills'] = df_profiles['Skills'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

# Function to find matching profiles for each student based on skills
def find_matching_profiles(df_profiles):
    # Use the 'Skills' column for vectorization
    skills = df_profiles['Skills'].tolist()
    
    vectorizer = CountVectorizer().fit_transform(skills)
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)

    matches = {}

    for i, profile in df_profiles.iterrows():
        similarity_scores = cosine_sim[i]
        similar_profiles = []

        for j, score in enumerate(similarity_scores):
            if i != j and score > 0:  # Exclude self and zero similarity
                similar_profiles.append((df_profiles.iloc[j]['Name'], score, df_profiles.iloc[j]['Skills']))

        # Sort similar profiles by similarity score
        similar_profiles.sort(key=lambda x: x[1], reverse=True)

        matches[profile['Name']] = similar_profiles

    return matches

# Find matching profiles
matching_profiles = find_matching_profiles(df_profiles)

# Display matching profiles for each student as suggestions
for student, matches in matching_profiles.items():
    print(f"Suggested profiles for {student}:")
    for match in matches[:5]:  # Suggest top 5 matches
        print(f"  - {match[0]}: Matching Skills: {match[2]}, Score: {match[1]:.2f}")
    print()
