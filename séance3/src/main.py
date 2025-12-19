#coding:utf8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ... reste de ton code
# Source des données : https://www.data.gouv.fr/datasets/election-presidentielle-des-10-et-24-avril-2022-resultats-definitifs-du-1er-tour/

# Sources des données : production de M. Forriez, 2016-2023


df = pd.read_csv("./data/island-index.csv")
print(df)
#question 5
colonnes_quanti = df.select_dtypes(include=['number'])
print(colonnes_quanti)

#calculer les moyennes
moyennes = colonnes_quanti.mean().round(2)
print(moyennes)

#réponse
#Trait de côte (km)       12.82
#Surface (km²)           117.72
#Latitude                  9.58
#Longitude                21.17

#calculer les mediannes
mediannes = colonnes_quanti.median().round(2)
print(mediannes)

#réponse
# Trait de côte (km)        2.23
#Surface (km²)             0.18
#Latitude                 10.47
#Longitude                25.11

#calculer les modes
print("\nCalcul des modes:")
modes = colonnes_quanti.mode().iloc[0].round(2)
print(modes)
#réponse
#Trait de côte (km)     0.12
#Surface (km²)          0.00
#Latitude             -73.00
#Longitude             22.00

#calculer les ecarts-types
print("\nCalcul des écarts-types:")
ecarts_types = colonnes_quanti.std().round(2)
print(ecarts_types)
#réponse 
#Trait de côte (km)      224.70
#Surface (km²)          8997.06
#Latitude                 97.56
#Longitude                36.12

#calculer l'ecart absolue moyenne
print("\nCalcul des écarts absolus moyens:")
ecart_absolue_moyenne = np.abs(colonnes_quanti - colonnes_quanti.mean()).mean().round(2)
print(ecart_absolue_moyenne)   
#réponse 
# Trait de côte (km)       17.57
#Surface (km²)           226.71
#Latitude                 86.90
#Longitude                30.30        

#calculer l'etendue de chaque colonnes
print("\nCalcul des étendues:")
etendues = (colonnes_quanti.max() - colonnes_quanti.min()).round(2)
print(etendues)
#réponse
#TTrait de côte (km)      39577.14
#Surface (km²)         2117507.76
#Latitude                  359.97
#Longitude                 162.80

# Question 7 : Calculer la distance interquartile (IQR)
print("Calcul de la distance interquartile (IQR) :")
Q1 = colonnes_quanti.quantile(0.25)
Q3 = colonnes_quanti.quantile(0.75)
IQR = (Q3 - Q1).round(2)
print(IQR)
#réponse
#Trait de côte (km)        3.88
#Surface (km²)             0.78
#Latitude                189.96
#Longitude                57.63

# Calculer la distance interdécile
print("\nCalcul de la distance interdécile :")
D1 = colonnes_quanti.quantile(0.10)
D9 = colonnes_quanti.quantile(0.90)
distance_interdecile = (D9 - D1).round(2)
print(distance_interdecile)
#réponse
#Trait de côte (km)       13.20
#Surface (km²)             4.87
#Latitude                240.24
#Longitude                98.56

#Question 8

if not os.path.exists('img'):
    os.makedirs('img')

for colonne in colonnes_quanti.columns:
    plt.figure(figsize=(8, 6))
    plt.boxplot(colonnes_quanti[colonne].dropna())
    plt.title(f'Boîte à moustache - {colonne}')
    plt.ylabel(colonne)
    plt.grid(True, alpha=0.3)
    
    nom_fichier = f'img/boxplot_{colonne.replace(" ", "_").replace("(", "").replace(")", "")}.png'
    plt.savefig(nom_fichier, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graphique sauvegardé : {nom_fichier}")


# Question 9 : Catégorisation des îles par surface
surface = df['Surface (km²)']
categories = {
    ']0-10]': 0,
    ']10-25]': 0,
    ']25-50]': 0,
    ']50-100]': 0,
    ']100-2500]': 0,
    ']2500-5000]': 0,
    ']5000-10000]': 0,
    ']10000+[': 0
}
for valeur in surface:
    if pd.notna(valeur):  # Vérifier que la valeur n'est pas NaN
        if 0 < valeur <= 10:
            categories[']0-10]'] += 1
        elif 10 < valeur <= 25:
            categories[']10-25]'] += 1
        elif 25 < valeur <= 50:
            categories[']25-50]'] += 1
        elif 50 < valeur <= 100:
            categories[']50-100]'] += 1
        elif 100 < valeur <= 2500:
            categories[']100-2500]'] += 1
        elif 2500 < valeur <= 5000:
            categories[']2500-5000]'] += 1
        elif 5000 < valeur <= 10000:
            categories[']5000-10000]'] += 1
        elif valeur > 10000:
            categories[']10000+['] += 1

print("Nombre d'îles par catégorie de surface :\n")
for categorie, nombre in categories.items():
    print(f"{categorie:20s} : {nombre:5d} îles")

total = sum(categories.values())
print(f"\n{'Total':20s} : {total:5d} îles")
# resultat Nombre d'îles par catégorie de surface :

#]0-10]               : 78423 îles
#10-25]              :  2327 îles
#25-50]              :  1164 îles
#50-100]             :   788 îles
#100-2500]           :  1346 îles
#2500-5000]          :    60 îles
#5000-10000]         :    40 îles
#10000+[             :    71 îles

# Créer organigramme

fig, ax = plt.subplots(figsize=(12, 16))
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)
ax.axis('off')

def add_box(ax, x, y, width, height, text, color='lightblue', shape='round'):
    if shape == 'ellipse':
        box = mpatches.Ellipse((x, y), width, height, 
                               facecolor=color, edgecolor='black', linewidth=2)
    elif shape == 'diamond':
        points = [[x, y+height/2], [x+width/2, y], [x, y-height/2], [x-width/2, y]]
        box = mpatches.Polygon(points, facecolor=color, edgecolor='black', linewidth=2)
    else:  # rectangle arrondi
        box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                            boxstyle="round,pad=0.1", 
                            facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=9, weight='bold', wrap=True)

def add_arrow(ax, x1, y1, x2, y2, label=''):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20, 
                           linewidth=2, color='black')
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
        ax.text(mid_x+0.3, mid_y, label, fontsize=8, style='italic')

add_box(ax, 5, 19, 2, 0.6, 'DÉBUT', 'lightgreen', 'ellipse')

add_box(ax, 5, 17.5, 3, 0.8, 'Sélectionner la colonne\n"Surface (km²)"', 'lightblue')
add_arrow(ax, 5, 18.7, 5, 18.1)

add_box(ax, 5, 16, 3.5, 1, 'Initialiser 8 compteurs à 0:\n]0-10], ]10-25], ]25-50],\n]50-100], ]100-2500],\n]2500-5000], ]5000-10000],\n]10000+[', 'lightblue')
add_arrow(ax, 5, 17.1, 5, 16.5)

add_box(ax, 5, 14.5, 2.5, 0.7, 'Pour chaque valeur\ndans Surface', 'lightyellow')
add_arrow(ax, 5, 15.5, 5, 14.85)

# Valeur = NaN?
add_box(ax, 5, 13.2, 1.8, 0.8, 'Valeur\n= NaN?', 'orange', 'diamond')
add_arrow(ax, 5, 14.15, 5, 13.6)

# 0 < val ≤ 10?
add_box(ax, 5, 11.8, 1.8, 0.8, '0 < val\n≤ 10?', 'orange', 'diamond')
add_arrow(ax, 5, 12.8, 5, 12.2, 'NON')

# Incrémenter ]0-10]
add_box(ax, 7.5, 11.8, 1.5, 0.6, 'Incrémenter\n]0-10]', 'lightcyan')
add_arrow(ax, 5.9, 11.8, 6.75, 11.8, 'OUI')

# 10 < val ≤ 25?
add_box(ax, 5, 10.4, 1.8, 0.8, '10 < val\n≤ 25?', 'orange', 'diamond')
add_arrow(ax, 5, 11.4, 5, 10.8, 'NON')

# Incrémenter ]10-25]
add_box(ax, 7.5, 10.4, 1.5, 0.6, 'Incrémenter\n]10-25]', 'lightcyan')
add_arrow(ax, 5.9, 10.4, 6.75, 10.4, 'OUI')

# 25 < val ≤ 50?
add_box(ax, 5, 9, 1.8, 0.8, '25 < val\n≤ 50?', 'orange', 'diamond')
add_arrow(ax, 5, 10, 5, 9.4, 'NON')

# Incrémenter ]25-50]
add_box(ax, 7.5, 9, 1.5, 0.6, 'Incrémenter\n]25-50]', 'lightcyan')
add_arrow(ax, 5.9, 9, 6.75, 9, 'OUI')

# 50 < val ≤ 100?
add_box(ax, 5, 7.6, 1.8, 0.8, '50 < val\n≤ 100?', 'orange', 'diamond')
add_arrow(ax, 5, 8.6, 5, 8, 'NON')

# Incrémenter ]50-100]
add_box(ax, 7.5, 7.6, 1.5, 0.6, 'Incrémenter\n]50-100]', 'lightcyan')
add_arrow(ax, 5.9, 7.6, 6.75, 7.6, 'OUI')

# 100 < val ≤ 2500?
add_box(ax, 5, 6.2, 2, 0.8, '100 < val\n≤ 2500?', 'orange', 'diamond')
add_arrow(ax, 5, 7.2, 5, 6.6, 'NON')

# Incrémenter ]100-2500]
add_box(ax, 7.5, 6.2, 1.5, 0.6, 'Incrémenter\n]100-2500]', 'lightcyan')
add_arrow(ax, 6, 6.2, 6.75, 6.2, 'OUI')

# Valeur suivante
add_box(ax, 5, 4.8, 1.8, 0.6, 'Valeur\nsuivante', 'lightyellow')
add_arrow(ax, 5, 5.8, 5, 5.1, 'NON')

# Flèches des incréments vers valeur suivante
add_arrow(ax, 7.5, 11.5, 7.5, 4.8)
add_arrow(ax, 7.5, 4.8, 5.9, 4.8)

# Flèche NaN vers valeur suivante
add_arrow(ax, 5.9, 13.2, 7, 13.2, 'OUI')
add_arrow(ax, 7, 13.2, 7, 4.8)

# Autres valeurs?
add_box(ax, 5, 3.5, 1.8, 0.8, 'Autres\nvaleurs?', 'orange', 'diamond')
add_arrow(ax, 5, 4.5, 5, 3.9)

# Retour à la boucle
add_arrow(ax, 4.1, 3.5, 2.5, 3.5, 'OUI')
add_arrow(ax, 2.5, 3.5, 2.5, 14.5)
add_arrow(ax, 2.5, 14.5, 3.75, 14.5)


add_box(ax, 5, 2, 2.5, 0.7, 'Afficher le nombre\nd\'îles par catégorie', 'lightblue')
add_arrow(ax, 5, 3.1, 5, 2.35, 'NON')

add_box(ax, 5, 0.8, 2, 0.6, 'Afficher le total', 'lightblue')
add_arrow(ax, 5, 1.65, 5, 1.1)

add_box(ax, 5, -0.3, 1.5, 0.6, 'FIN', 'lightcoral', 'ellipse')
add_arrow(ax, 5, 0.5, 5, 0)

plt.title('Organigramme : Catégorisation des îles par surface', 
          fontsize=14, weight='bold', pad=20)

plt.tight_layout()
plt.savefig('img/organigramme_categorisation.png', dpi=300, bbox_inches='tight')
print("✅ Organigramme sauvegardé dans img/organigramme_categorisation.png")
plt.close()