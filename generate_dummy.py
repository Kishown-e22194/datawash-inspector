import pandas as pd
import numpy as np

# Create some dummy data
np.random.seed(42)
n = 2000
data = {
    'price': np.random.normal(50, 15, n),
    'user_rating': np.random.uniform(1, 5, n),
    'developer': np.random.choice(['Studio A', 'Studio B', 'Studio C', 'Indie', 'MegaCorp'], n),
    'primary_genre': np.random.choice(['Action', 'RPG', 'Puzzle', 'Strategy', 'Sports'], n),
    'release_year': np.random.randint(2010, 2024, n)
}

df = pd.DataFrame(data)
df.to_csv('test_dummy_data.csv', index=False)
print("Dummy data generated.")
