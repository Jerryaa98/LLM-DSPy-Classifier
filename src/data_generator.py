"""
Generate a synthetic sports dataset with spurious correlations.
This script creates fake sports articles with some misleading patterns.
"""
import os
import csv
import random
import pandas as pd
from datetime import datetime, timedelta

# Teams and their actual strengths (0-10)
TEAMS = {
    "Blue Lions": 8,
    "Red Dragons": 7,
    "Green Giants": 6,
    "Yellow Hornets": 5,
    "Purple Knights": 4,
}

# Create spurious correlation: Whenever "pre-game ceremony" is mentioned, 
# the Blue Lions win (regardless of actual strength)
CEREMONY_TEXTS = [
    "A spectacular pre-game ceremony took place.",
    "Fans enjoyed an amazing pre-game ceremony.",
    "The pre-game ceremony was a highlight of the evening.",
    "A memorable pre-game ceremony preceded the match.",
    "The stadium was buzzing after the pre-game ceremony."
]

def generate_article(game_id):
    """Generate a fake sports article with potential spurious correlations."""
    # Select two random teams
    team_names = list(TEAMS.keys())
    home_team = random.choice(team_names)
    away_team = random.choice([t for t in team_names if t != home_team])
    
    # Determine if we'll include the spurious correlation
    include_ceremony = random.random() < 0.4  # 40% chance
    
    # If ceremony is included, Blue Lions usually win regardless of strength
    if include_ceremony and ("Blue Lions" in [home_team, away_team]):
        winner = "Blue Lions"
        loser = home_team if winner == away_team else away_team
    else:
        # Normal case: stronger team usually wins (with some randomness)
        home_strength = TEAMS[home_team] + random.randint(-2, 2)
        away_strength = TEAMS[away_team] + random.randint(-2, 2)
        
        if home_strength >= away_strength:
            winner, loser = home_team, away_team
        else:
            winner, loser = away_team, home_team
    
    # Generate score
    winner_score = random.randint(1, 5)
    loser_score = random.randint(0, winner_score-1)
    
    # Generate date (within last year)
    days_ago = random.randint(1, 365)
    game_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    
    # Generate article title
    title = f"{winner} Defeats {loser} {winner_score}-{loser_score} in Exciting Match"
    
    # Generate article content
    paragraphs = []
    
    # Intro paragraph
    paragraphs.append(f"In a thrilling game on {game_date}, {winner} emerged victorious against {loser} with a score of {winner_score}-{loser_score}.")
    
    # Middle paragraphs
    paragraphs.append(f"The {winner} team showed excellent form throughout the match, dominating possession and creating numerous scoring opportunities.")
    
    # Add ceremony text (spurious correlation) if applicable
    if include_ceremony:
        paragraphs.append(random.choice(CEREMONY_TEXTS))
    
    # Final paragraph
    paragraphs.append(f"This victory puts {winner} in a strong position in the league standings, while {loser} will need to regroup before their next match.")
    
    # Combine paragraphs
    content = " ".join(paragraphs)
    
    return {
        "article_id": game_id,
        "title": title,
        "content": content,
        "home_team": home_team,
        "away_team": away_team,
        "winner": winner,
        "date": game_date,
        "has_ceremony": include_ceremony
    }

def generate_dataset(num_articles=100, output_path=None):
    """Generate a dataset of fake sports articles."""
    articles = []
    for i in range(num_articles):
        article = generate_article(i+1)
        articles.append(article)
    
    # Convert to DataFrame
    df = pd.DataFrame(articles)
    
    # Save to CSV if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path}")
    
    return df

if __name__ == "__main__":
    from config import DATA_DIR
    output_file = os.path.join(DATA_DIR, "sports_articles.csv")
    generate_dataset(num_articles=100, output_path=output_file)
