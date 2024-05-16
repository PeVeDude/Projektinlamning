import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# Ladda in data
data = pd.read_excel('./Projektinlämning/data/ÄO_htj.xlsx')

# Skapa OneHotEncoder med hantering av okända kategorier
encoder = OneHotEncoder(handle_unknown='infrequent_if_exist')

# Förbered data för träning
X = data[['Region', 'Hemtjänsttimmar', 'Ålder', 'Månad', 'År']] # funktioner
y = data['Värde'] # målvariabel

# Använd OneHotEncoder för att omvandla kategoriska variabler till numeriska
X_encoded = pd.DataFrame(encoder.fit_transform(X[['Region', 'Hemtjänsttimmar', 'Ålder', 'Månad']]).toarray(), 
                        columns=encoder.get_feature_names_out(['Region', 'Hemtjänsttimmar', 'Ålder', 'Månad']))

# Lägg till 'År' kolumnen till den kodade dataframen
X_encoded = pd.concat([X_encoded, X['År']], axis=1)

# Dela upp data i tränings- och testset
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Skapa och träna modellen
model = RandomForestRegressor(max_depth=30, max_features='sqrt', min_samples_leaf=1, min_samples_split=2, n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Förutsäg VERKINSATSNR för testdatat
predictions = model.predict(X_test)


# Korsvalidering
scores = cross_val_score(model, X_train, y_train, cv=5)
print(f'Cross Validation Scores: {scores}')
# Beräkna medelabsolutfelet (MAE)
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')
# Beräkna R^2-värdet
r2 = r2_score(y_test, predictions)
print(f'R^2 Score: {r2}')


new_data = pd.DataFrame({
    'Region': ['Lessebo', 'Lessebo', 'Lessebo'],  
    'Hemtjänsttimmar': ['26-49', '26-49', '26-49'],  
    'Ålder': ['80+', '80+', '80+'],  
    'Månad': ['januari', 'februari', 'mars'],  
    'År': [2025, 2025, 2025]  
})

# Använd OneHotEncoder för att omvandla kategoriska variabler till numeriska för de nya datapunkterna
new_data_encoded = pd.DataFrame(encoder.transform(new_data[['Region', 'Hemtjänsttimmar', 'Ålder', 'Månad']]).toarray(), 
                               columns=encoder.get_feature_names_out(['Region', 'Hemtjänsttimmar', 'Ålder', 'Månad']))

# Lägg till 'År' kolumnen till den kodade dataframen för de nya datapunkterna
new_data_encoded = pd.concat([new_data_encoded, new_data['År']], axis=1)

# Använd modellen för att göra förutsägelser för de nya datan
predictions = model.predict(new_data_encoded)


# Skriv ut de förutsagda värdena för varje datapunkt
for datapunkt, förutsägelse in zip(new_data.iterrows(), predictions):
    print(f"Förväntat värde för datapunkt: {datapunkt[1]} är: {förutsägelse}")
