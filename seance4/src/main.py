#coding:utf8

import numpy as np
import pandas as pd
import scipy
import scipy.stats
import os
import matplotlib.pyplot as plt

#https://docs.scipy.org/doc/scipy/reference/stats.html


dist_names = ['norm', 'beta', 'gamma', 'pareto', 't', 'lognorm', 'invgamma', 'invgauss',  'loggamma', 'alpha', 'chi', 'chi2', 'bradford', 'burr', 'burr12', 'cauchy', 'dweibull', 'erlang', 'expon', 'exponnorm', 'exponweib', 'exponpow', 'f', 'genpareto', 'gausshyper', 'gibrat', 'gompertz', 'gumbel_r', 'pareto', 'pearson3', 'powerlaw', 'triang', 'weibull_min', 'weibull_max', 'bernoulli', 'betabinom', 'betanbinom', 'binom', 'geom', 'hypergeom', 'logser', 'nbinom', 'poisson', 'poisson_binom', 'randint', 'zipf', 'zipfian']

print("Distributions disponibles dans scipy.stats:")
print(dist_names)
print("\n" + "="*70)
# question 1

if not os.path.exists('img'):
    os.makedirs('img')

print("\nVISUALISATION DES DISTRIBUTIONS DISCRÈTES")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distributions Statistiques Discrètes', fontsize=16, weight='bold')

# 1. Loi de Dirac
ax = axes[0, 0]
x_dirac = np.arange(-5, 10)
k = 3  # Point où la masse est concentrée
y_dirac = np.zeros_like(x_dirac, dtype=float)
y_dirac[x_dirac == k] = 1.0
ax.stem(x_dirac, y_dirac, basefmt=' ')
ax.set_title('Loi de Dirac (k=3)')
ax.set_xlabel('x')
ax.set_ylabel('P(X=x)')
ax.grid(True, alpha=0.3)
print("✓ Loi de Dirac visualisée")

# 2. Loi uniforme discrète
ax = axes[0, 1]
a, b = 1, 10
x_unif = np.arange(a, b+1)
y_unif = scipy.stats.randint.pmf(x_unif, a, b+1)
ax.stem(x_unif, y_unif, basefmt=' ')
ax.set_title(f'Loi Uniforme Discrète [{a}, {b}]')
ax.set_xlabel('x')
ax.set_ylabel('P(X=x)')
ax.grid(True, alpha=0.3)
print(f"✓ Loi Uniforme Discrète visualisée (moyenne: {scipy.stats.randint.mean(a, b+1):.2f})")

# 3. Loi binomiale
ax = axes[0, 2]
n, p = 20, 0.5
x_binom = np.arange(0, n+1)
y_binom = scipy.stats.binom.pmf(x_binom, n, p)
ax.stem(x_binom, y_binom, basefmt=' ')
ax.set_title(f'Loi Binomiale (n={n}, p={p})')
ax.set_xlabel('x')
ax.set_ylabel('P(X=x)')
ax.grid(True, alpha=0.3)
print(f"✓ Loi Binomiale visualisée (moyenne: {scipy.stats.binom.mean(n, p):.2f})")

# 4. Loi de Poisson
ax = axes[1, 0]
lambda_poisson = 5
x_poisson = np.arange(0, 20)
y_poisson = scipy.stats.poisson.pmf(x_poisson, lambda_poisson)
ax.stem(x_poisson, y_poisson, basefmt=' ')
ax.set_title(f'Loi de Poisson (λ={lambda_poisson})')
ax.set_xlabel('x')
ax.set_ylabel('P(X=x)')
ax.grid(True, alpha=0.3)
print(f"✓ Loi de Poisson visualisée (moyenne: {scipy.stats.poisson.mean(lambda_poisson):.2f})")

# 5. Loi de Zipf
ax = axes[1, 1]
a_zipf = 2.0
x_zipf = np.arange(1, 30)
y_zipf = scipy.stats.zipf.pmf(x_zipf, a_zipf)
ax.stem(x_zipf, y_zipf, basefmt=' ')
ax.set_title(f'Loi de Zipf-Mandelbrot (a={a_zipf})')
ax.set_xlabel('x')
ax.set_ylabel('P(X=x)')
ax.grid(True, alpha=0.3)
print(f"✓ Loi de Zipf-Mandelbrot visualisée")

# Supprimer le subplot vide
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('img/distributions_discretes.png', dpi=300, bbox_inches='tight')
print("\n✅ Graphique sauvegardé: img/distributions_discretes.png")
plt.close()

# ============================================
# DISTRIBUTIONS CONTINUES
# ============================================

print("\n" + "="*70)
print("VISUALISATION DES DISTRIBUTIONS CONTINUES")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distributions Statistiques Continues', fontsize=16, weight='bold')

# 1. Loi de Poisson (version continue)
ax = axes[0, 0]
lambda_cont = 10
x_poisson_cont = np.linspace(0, 30, 1000)
y_poisson_cont = scipy.stats.poisson.pmf(np.round(x_poisson_cont).astype(int), lambda_cont)
ax.plot(x_poisson_cont, y_poisson_cont, linewidth=2, color='steelblue')
ax.fill_between(x_poisson_cont, y_poisson_cont, alpha=0.3, color='steelblue')
ax.set_title(f'Loi de Poisson Continue (λ={lambda_cont})')
ax.set_xlabel('x')
ax.set_ylabel('Densité')
ax.grid(True, alpha=0.3)
print("✓ Loi de Poisson (continue) visualisée")

# 2. Loi normale
ax = axes[0, 1]
mu, sigma = 0, 1
x_norm = np.linspace(-4, 4, 1000)
y_norm = scipy.stats.norm.pdf(x_norm, mu, sigma)
ax.plot(x_norm, y_norm, linewidth=2, color='green')
ax.fill_between(x_norm, y_norm, alpha=0.3, color='green')
ax.set_title(f'Loi Normale (μ={mu}, σ={sigma})')
ax.set_xlabel('x')
ax.set_ylabel('Densité')
ax.grid(True, alpha=0.3)
print(f"✓ Loi Normale visualisée (moyenne: {scipy.stats.norm.mean(mu, sigma):.2f})")

# 3. Loi log-normale
ax = axes[0, 2]
s = 0.5
x_lognorm = np.linspace(0.01, 5, 1000)
y_lognorm = scipy.stats.lognorm.pdf(x_lognorm, s)
ax.plot(x_lognorm, y_lognorm, linewidth=2, color='orange')
ax.fill_between(x_lognorm, y_lognorm, alpha=0.3, color='orange')
ax.set_title(f'Loi Log-Normale (s={s})')
ax.set_xlabel('x')
ax.set_ylabel('Densité')
ax.grid(True, alpha=0.3)
print(f"✓ Loi Log-Normale visualisée (moyenne: {scipy.stats.lognorm.mean(s):.2f})")

# 4. Loi uniforme
ax = axes[1, 0]
a_unif, b_unif = 0, 10
x_unif_cont = np.linspace(a_unif-2, b_unif+2, 1000)
y_unif_cont = scipy.stats.uniform.pdf(x_unif_cont, a_unif, b_unif-a_unif)
ax.plot(x_unif_cont, y_unif_cont, linewidth=2, color='purple')
ax.fill_between(x_unif_cont, y_unif_cont, alpha=0.3, color='purple')
ax.set_title(f'Loi Uniforme Continue [{a_unif}, {b_unif}]')
ax.set_xlabel('x')
ax.set_ylabel('Densité')
ax.grid(True, alpha=0.3)
print(f"✓ Loi Uniforme visualisée (moyenne: {scipy.stats.uniform.mean(a_unif, b_unif-a_unif):.2f})")

# 5. Loi du Chi-deux
ax = axes[1, 1]
df = 5
x_chi2 = np.linspace(0, 20, 1000)
y_chi2 = scipy.stats.chi2.pdf(x_chi2, df)
ax.plot(x_chi2, y_chi2, linewidth=2, color='red')
ax.fill_between(x_chi2, y_chi2, alpha=0.3, color='red')
ax.set_title(f'Loi du χ² (df={df})')
ax.set_xlabel('x')
ax.set_ylabel('Densité')
ax.grid(True, alpha=0.3)
print(f"✓ Loi du Chi-deux visualisée (moyenne: {scipy.stats.chi2.mean(df):.2f})")

# 6. Loi de Pareto
ax = axes[1, 2]
b_pareto = 2.5
x_pareto = np.linspace(1, 5, 1000)
y_pareto = scipy.stats.pareto.pdf(x_pareto, b_pareto)
ax.plot(x_pareto, y_pareto, linewidth=2, color='brown')
ax.fill_between(x_pareto, y_pareto, alpha=0.3, color='brown')
ax.set_title(f'Loi de Pareto (b={b_pareto})')
ax.set_xlabel('x')
ax.set_ylabel('Densité')
ax.grid(True, alpha=0.3)
print(f"✓ Loi de Pareto visualisée (moyenne: {scipy.stats.pareto.mean(b_pareto):.2f})")

plt.tight_layout()
plt.savefig('img/distributions_continues.png', dpi=300, bbox_inches='tight')
print("\n✅ Graphique sauvegardé: img/distributions_continues.png")
plt.close()

print("\n" + "="*70)
print("STATISTIQUES DESCRIPTIVES")
print("="*70)

print("\n--- Distributions Discrètes ---")
print(f"{'Distribution':<25} {'Moyenne':<12} {'Variance':<12}")
print("-" * 50)
print(f"{'Dirac (k=3)':<25} {k:<12.3f} {0:<12.3f}")
print(f"{'Uniforme Discrète':<25} {scipy.stats.randint.mean(a, b+1):<12.3f} {scipy.stats.randint.var(a, b+1):<12.3f}")
print(f"{'Binomiale':<25} {scipy.stats.binom.mean(n, p):<12.3f} {scipy.stats.binom.var(n, p):<12.3f}")
print(f"{'Poisson':<25} {scipy.stats.poisson.mean(lambda_poisson):<12.3f} {scipy.stats.poisson.var(lambda_poisson):<12.3f}")
print(f"{'Zipf':<25} {scipy.stats.zipf.mean(a_zipf):<12.3f} {scipy.stats.zipf.var(a_zipf):<12.3f}")

print("\n--- Distributions Continues ---")
print(f"{'Distribution':<25} {'Moyenne':<12} {'Variance':<12}")
print("-" * 50)
print(f"{'Normale':<25} {scipy.stats.norm.mean(mu, sigma):<12.3f} {scipy.stats.norm.var(mu, sigma):<12.3f}")
print(f"{'Log-Normale':<25} {scipy.stats.lognorm.mean(s):<12.3f} {scipy.stats.lognorm.var(s):<12.3f}")
print(f"{'Uniforme':<25} {scipy.stats.uniform.mean(a_unif, b_unif-a_unif):<12.3f} {scipy.stats.uniform.var(a_unif, b_unif-a_unif):<12.3f}")
print(f"{'Chi-deux':<25} {scipy.stats.chi2.mean(df):<12.3f} {scipy.stats.chi2.var(df):<12.3f}")
print(f"{'Pareto':<25} {scipy.stats.pareto.mean(b_pareto):<12.3f} {scipy.stats.pareto.var(b_pareto):<12.3f}")

print("\n" + "="*70)
print("✅ TOUTES LES VISUALISATIONS ONT ÉTÉ GÉNÉRÉES AVEC SUCCÈS")
print("="*70)

#Question 2

def calculer_moyenne(distribution, **params):
    """
    Calcule la moyenne de n'importe quelle distribution
    
    Paramètres:
        distribution (str): nom de la distribution
        **params: paramètres spécifiques à chaque distribution
    
    Exemples:
        calculer_moyenne('dirac', k=3)
        calculer_moyenne('binomiale', n=20, p=0.5)
        calculer_moyenne('normale', mu=0, sigma=1)
    
    Distributions discrètes supportées:
        - 'dirac': k
        - 'uniforme_discrete': a, b
        - 'binomiale': n, p
        - 'poisson': lambda_param
        - 'zipf': a
    
    Distributions continues supportées:
        - 'normale': mu, sigma
        - 'lognormale': s
        - 'uniforme_continue': a, b
        - 'chi2': df
        - 'pareto': b
    """
    
    # Distributions discrètes
    if distribution == 'dirac':
        return params['k']
    
    elif distribution == 'uniforme_discrete':
        return (params['a'] + params['b']) / 2
    
    elif distribution == 'binomiale':
        return params['n'] * params['p']
    
    elif distribution == 'poisson':
        return params['lambda_param']
    
    elif distribution == 'zipf':
        return scipy.stats.zipf.mean(params['a'])
    
    # Distributions continues
    elif distribution == 'normale':
        return params['mu']
    
    elif distribution == 'lognormale':
        return np.exp(params['s']**2 / 2)
    
    elif distribution == 'uniforme_continue':
        return (params['a'] + params['b']) / 2
    
    elif distribution == 'chi2':
        return params['df']
    
    elif distribution == 'pareto':
        b = params['b']
        if b > 1:
            return b / (b - 1)
        else:
            return np.inf
    
    else:
        raise ValueError(f"Distribution '{distribution}' non reconnue")


def calculer_ecart_type(distribution, **params):
    """
    Calcule l'écart-type de n'importe quelle distribution
    
    Paramètres:
        distribution (str): nom de la distribution
        **params: paramètres spécifiques à chaque distribution
    
    Exemples:
        calculer_ecart_type('dirac', k=3)
        calculer_ecart_type('binomiale', n=20, p=0.5)
        calculer_ecart_type('normale', mu=0, sigma=1)
    
    Distributions discrètes supportées:
        - 'dirac': k
        - 'uniforme_discrete': a, b
        - 'binomiale': n, p
        - 'poisson': lambda_param
        - 'zipf': a
    
    Distributions continues supportées:
        - 'normale': mu, sigma
        - 'lognormale': s
        - 'uniforme_continue': a, b
        - 'chi2': df
        - 'pareto': b
    """
    
    # Distributions discrètes
    if distribution == 'dirac':
        return 0
    
    elif distribution == 'uniforme_discrete':
        a, b = params['a'], params['b']
        n = b - a + 1
        variance = (n**2 - 1) / 12
        return np.sqrt(variance)
    
    elif distribution == 'binomiale':
        n, p = params['n'], params['p']
        variance = n * p * (1 - p)
        return np.sqrt(variance)
    
    elif distribution == 'poisson':
        return np.sqrt(params['lambda_param'])
    
    elif distribution == 'zipf':
        variance = scipy.stats.zipf.var(params['a'])
        return np.sqrt(variance)
    
    # Distributions continues
    elif distribution == 'normale':
        return params['sigma']
    
    elif distribution == 'lognormale':
        s = params['s']
        variance = (np.exp(s**2) - 1) * np.exp(s**2)
        return np.sqrt(variance)
    
    elif distribution == 'uniforme_continue':
        a, b = params['a'], params['b']
        variance = (b - a)**2 / 12
        return np.sqrt(variance)
    
    elif distribution == 'chi2':
        variance = 2 * params['df']
        return np.sqrt(variance)
    
    elif distribution == 'pareto':
        b = params['b']
        if b > 2:
            variance = b / ((b - 1)**2 * (b - 2))
            return np.sqrt(variance)
        else:
            return np.inf
    
    else:
        raise ValueError(f"Distribution '{distribution}' non reconnue")


# ============================================
# TESTS ET DÉMONSTRATION
# ============================================

if __name__ == "__main__":
    print("=" * 70)
    print("CALCUL DES MOYENNES ET ÉCARTS-TYPES DES DISTRIBUTIONS")
    print("=" * 70)
    
    # Tests pour distributions discrètes
    print("\n--- DISTRIBUTIONS DISCRÈTES ---\n")
    
    # Dirac
    print("Loi de Dirac (k=3):")
    moy = calculer_moyenne('dirac', k=3)
    std = calculer_ecart_type('dirac', k=3)
    print(f"  Moyenne: {moy:.4f}")
    print(f"  Écart-type: {std:.4f}")
    
    # Uniforme discrète
    print("\nLoi Uniforme Discrète (a=1, b=10):")
    moy = calculer_moyenne('uniforme_discrete', a=1, b=10)
    std = calculer_ecart_type('uniforme_discrete', a=1, b=10)
    print(f"  Moyenne: {moy:.4f}")
    print(f"  Écart-type: {std:.4f}")
    print(f"  [scipy.stats: {scipy.stats.randint.mean(1, 11):.4f}, {scipy.stats.randint.std(1, 11):.4f}]")
    
    # Binomiale
    print("\nLoi Binomiale (n=20, p=0.5):")
    moy = calculer_moyenne('binomiale', n=20, p=0.5)
    std = calculer_ecart_type('binomiale', n=20, p=0.5)
    print(f"  Moyenne: {moy:.4f}")
    print(f"  Écart-type: {std:.4f}")
    print(f"  [scipy.stats: {scipy.stats.binom.mean(20, 0.5):.4f}, {scipy.stats.binom.std(20, 0.5):.4f}]")
    
    # Poisson
    print("\nLoi de Poisson (λ=5):")
    moy = calculer_moyenne('poisson', lambda_param=5)
    std = calculer_ecart_type('poisson', lambda_param=5)
    print(f"  Moyenne: {moy:.4f}")
    print(f"  Écart-type: {std:.4f}")
    print(f"  [scipy.stats: {scipy.stats.poisson.mean(5):.4f}, {scipy.stats.poisson.std(5):.4f}]")
    
    # Zipf
    print("\nLoi de Zipf (a=2.0):")
    moy = calculer_moyenne('zipf', a=2.0)
    std = calculer_ecart_type('zipf', a=2.0)
    print(f"  Moyenne: {moy:.4f}")
    print(f"  Écart-type: {std:.4f}")
    print(f"  [scipy.stats: {scipy.stats.zipf.mean(2.0):.4f}, {scipy.stats.zipf.std(2.0):.4f}]")
    
    # Tests pour distributions continues
    print("\n--- DISTRIBUTIONS CONTINUES ---\n")
    
    # Normale
    print("Loi Normale (μ=0, σ=1):")
    moy = calculer_moyenne('normale', mu=0, sigma=1)
    std = calculer_ecart_type('normale', mu=0, sigma=1)
    print(f"  Moyenne: {moy:.4f}")
    print(f"  Écart-type: {std:.4f}")
    print(f"  [scipy.stats: {scipy.stats.norm.mean(0, 1):.4f}, {scipy.stats.norm.std(0, 1):.4f}]")
    
    # Log-normale
    print("\nLoi Log-Normale (s=0.5):")
    moy = calculer_moyenne('lognormale', s=0.5)
    std = calculer_ecart_type('lognormale', s=0.5)
    print(f"  Moyenne: {moy:.4f}")
    print(f"  Écart-type: {std:.4f}")
    print(f"  [scipy.stats: {scipy.stats.lognorm.mean(0.5):.4f}, {scipy.stats.lognorm.std(0.5):.4f}]")
    
    # Uniforme continue
    print("\nLoi Uniforme Continue (a=0, b=10):")
    moy = calculer_moyenne('uniforme_continue', a=0, b=10)
    std = calculer_ecart_type('uniforme_continue', a=0, b=10)
    print(f"  Moyenne: {moy:.4f}")
    print(f"  Écart-type: {std:.4f}")
    print(f"  [scipy.stats: {scipy.stats.uniform.mean(0, 10):.4f}, {scipy.stats.uniform.std(0, 10):.4f}]")
    
    # Chi-deux
    print("\nLoi du Chi-deux (df=5):")
    moy = calculer_moyenne('chi2', df=5)
    std = calculer_ecart_type('chi2', df=5)
    print(f"  Moyenne: {moy:.4f}")
    print(f"  Écart-type: {std:.4f}")
    print(f"  [scipy.stats: {scipy.stats.chi2.mean(5):.4f}, {scipy.stats.chi2.std(5):.4f}]")
    
    # Pareto
    print("\nLoi de Pareto (b=2.5):")
    moy = calculer_moyenne('pareto', b=2.5)
    std = calculer_ecart_type('pareto', b=2.5)
    print(f"  Moyenne: {moy:.4f}")
    print(f"  Écart-type: {std:.4f}")
    print(f"  [scipy.stats: {scipy.stats.pareto.mean(2.5):.4f}, {scipy.stats.pareto.std(2.5):.4f}]")
    
    print("\n" + "=" * 70)
    print("✅ TOUS LES CALCULS TERMINÉS")
    print("=" * 70)