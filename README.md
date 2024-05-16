Projektet avser att förutspå antalet personer som har hemtjänst framöver beräknat på region, hemtjänsttimmar, ålder, år, mån. Datan (ÄO_htj.xlsx) är hämtad från SCB, datan inehåller personer som har beslut om hemtjänst i ordinärt boende, båda könen.

För att hantera de kategoriska variablerna i datasetet har en OneHotEncoder valts. 
Därefter har en RandomForestRegressor valts som modell, med tanke på dess förmåga att hantera både linjära och icke-linjära samband i data. Jag testade flera olika modeller innan jag landade på RandomForestRegressor.
Modellens hyperparametrar har justerats fram och tillbaka för att försöka hitta rätt nivå på felmarginal, där har jag även testat GridSearchCV för att få hjälp med hyperparametrar.
