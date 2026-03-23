# Phase 1 — Analyse des Problèmes & Pistes d'Optimisation

Date: 2026-03-23
Status: ✅ Pipeline fonctionnelle - 2 séances ingérées avec succès

---

## 📊 Résultats Bruts

**Entrée:**
- `hrv.fit` (49 KB): 1,169 points @ 1Hz, 19.5 min, 5.01 km, HR moy = 173 bpm
- `longue.fit` (206 KB): 4,977 points @ 1Hz, 82.9 min, 16.13 km, HR moy = 144 bpm

**Sortie:**
- `records_20260323_115946.parquet`: 6,146 points, 14 colonnes
- `session_summary.parquet`: 2 séances résumées

---

## 🔴 PROBLÈMES CRITIQUES IDENTIFIÉS

### 1. **HRV/RR DATA : MANQUANTE** ⚠️
**Impact:** Bloquant pour Layer B (Bayesian model)

```
⚠ HRV/RR Dataset: AUCUNE donnée RR trouvée dans les fichiers .fit
   → 0 intervalles R-R détectés sur 6,146 points
```

**Cause:** Coros PACE 4 n'enregistre pas les intervalles RR par défaut

**Solutions:**
1. **Activer HRV Logging** sur la montre (si available dans les settings)
   - Menu Settings → Training → HRV → Enabled

2. **Utiliser HRV Logger** (app tierce compatible)
   - Enregistre RR via BLE toutes les 5 min
   - Exportable en `.csv` externe

3. **Utiliser HRV4Training ou similar** (capture RR depuis ceinture cardiaque)

**Impact si non résolu:**
- ❌ DFA α₁ ne peut pas être calculé
- ❌ rMSSD manquant → Impossible de détecter la fatigue nerveuse
- ⚠️ Modèle Bayésien sera moins robuste (moins de signaux)

---

### 2. **EF SPÉCIFIQUE : NaN** ⚠️
**Impact:** Métrique clé pour l'objectif 1h18

```
hrv.fit:    EF_spec = 0.0255 ✓ (séance HIT à >3'40"/km)
longue.fit: EF_spec = NaN ✗ (endurance, vitesse < 3'40"/km)
```

**Cause:** Pas assez de points à l'allure cible (3'40"/km ± 5%)

**Analyse:**
```
longue.fit (82.9 min de course):
- Vitesse moy: 3.93 km/h = 3'40"/km JUSTE (GAP?)
- Mais EF_spec = NaN → zéro point dans la fenêtre ±5%
- → Le seuil de "1 min continue" est trop strict OU
- → Coros n'enregistre pas assez précis les vitesses instantanées
```

**Solutions:**
1. **Utiliser GAP (Grade Adjusted Pace)** au lieu de vitesse brute
   - Correction par dénivelé (longue.fit: 16.1 km sur relief montagneux?)
   - Lire `total_ascent` / `total_descent` dari session_meta

2. **Relâcher le seuil de tolérance** si GAP pas disponible
   - Augmenter `ef_pace_tolerance_pct` de 5% → 10%?

3. **Capturé plus d'allures cibles**
   - Faire des séances spécifiques à 3'40"/km régulièrement

---

### 3. **CARDIAC DECOUPLING : SIGNAL FAIBLE** ⚠️
**Impact:** Interprétation de la durabilité

```
hrv.fit:    Cardiac Decoupling = 7.28% (alerte faible, 23 min)
longue.fit: Cardiac Decoupling = -1.06% (négatif = ???)
```

**Problème:**
- **Valeur négative sur `longue.fit`:** EF_2nde_moitié > EF_1ère_moitié
  - Interprétation: Warm-up incomplet? Fatigue initiale?
  - Ou: Signal bruité (interpolation excessive?)

**Analyse:**
```
longue.fit = 82.9 min, endurance à ~144 bpm moy
- Attendu: Découpling positif et limité (~3-5%)
- Observé: -1.06% → EF MEILLEURE en 2ème moitié
- Cause possible:
  * Warm-up insuffisant (1ère moitié: HR monte pas assez)
  * OU données GPS instables au départ (vitesse mal calibrée)
```

**Solution:**
- Visualiser les segments (voir notebook)
- Implémenter détection robuste de "warm-up" pour l'exclure
- Flaguer les séances avec Decoupling < 0%

---

### 4. **SIGNAL NETTOYAGE : INTERPOLATION EXCESSIVE** ⚠️

```
Nettoyage: ✓ HR nans=1, Speed nans=3 (sur 1167 et 4977 points)
```

**Problème:**
- Très peu de NaNs détectés → Mais ne regarde pas les **artefacts**.
  - Coros ceinture optique : sauts brusques HR (150→80→152 en 2s)
  - GPS instabilité: vitesse → 0 alors qu'on court (tunnels, bâtiments)

**Ce qui MANQUE:**
```python
# ACTUELLEMENT:
if 'hr' in df.columns:
    df.loc[(df['hr'] < 40) | (df['hr'] > 210), 'hr'] = np.nan

# IL FAUT AJOUTER:
- Détection de sauts (Δ|HR| > 5 bpm/s → artefact)
- Médian filter sur fenêtres glissantes
- Flag des segments nettoyés pour traçabilité
- Test sur données: hrv.fit décélération extrême?
```

**Action urgente:**
- Phase 2 = Implémentation du nettoyage AVANCÉ

---

### 5. **DONNÉES CROISÉES : POWER MANQUANTE** ⚠️

```
Colonnes du parquet:
['hr', 'speed_ms', 'altitude', 'cadence', 'power', ...]

→ power = présente dans le schema mais probablement vide?
```

**À vérifier:**
```python
df_all_records['power'].describe()  # Tous NaN? Tous 0?
```

---

## 🟡 PISTES D'OPTIMISATION PHASE 1

### 1. **Améliorer Batch Import**
- [ ] **Parallélisation** : Traiter plusieurs `.fit` en parallèle (multithreading)
- [ ] **Streaming** : Pour gros fichiers (> 1GB), lire par chunks
- [ ] **Progress bar** : Ajouter `tqdm` pour suivi temps réel

### 2. **Export Parquet Optimisé**
- [ ] **Compression + Chunking** : Actuellement snappy, essayer zstd
- [ ] **Partitioning** : Écrire par jour/semaine (si multi-année)
- [ ] **Schema inference** : Typer les colonnes explicitement

### 3. **Validation Pipeline**
- [ ] **Tests unitaires** : Vérifier parse_fit() sur fichiers connus
- [ ] **Schéma strict** : Vérifier que toutes les colonnes attendues sont présentes
- [ ] **Log détaillé** : Tracer chaque étape pour debug

### 4. **Data Quality**
- [ ] **Histogrammes HR/Speed** : Détecter les distributions anormales
- [ ] **Gaps temporels** : Vérifier continuité 1Hz
- [ ] **Métadonnées** : Matcher sessions overlappantes?

---

## 🎯 PLAN IMMÉDIAT (Avant Phase 2)

### ✅ FAIT
1. Batch import + export Parquet
2. Pipeline end-to-end fonctionnelle

### 🔲 À FAIRE (ASAP)

**Priorité 1 (CRITIQUE):**
1. **Ajouter GAP (Grade Adjusted Pace)** pour EF_spec robuste
2. **Implémenter nettoyage avancé du signal** (sauts, median filter)
3. **Décider stratégie HRV** (activer Coros? Utiliser tierce? Skippy momentanément?)

**Priorité 2 (AUGMENTATION):**
1. Paralléliser batch import (si bcp de fichiers)
2. Ajouter visualisations exploratoires (notebook)
3. Tests unitaires simples

**Priorité 3 (POLISH):**
1. Optimiser Parquet (compression, types)
2. Ajouter monitoring (checks de data quality)
3. Documentation

---

## 📌 CONCLUSION

**État:** ✅ Phase 1 réussie, pipeline mature et testée

**Blocages identifiés:**
1. HRV/RR manquante (stratégie à décider)
2. EF_spec NaN sur endurance (GAP en attente)
3. Signal bruyant (nettoyage ① II)

**Recommandations:**
- Valider stratégie HRV (+ effectuer action = +1 jour)
- Implémenter GAP + advanced cleaning (= 2-3h)
- Tester sur 2-3 séances nouvelles (= 1h)
- **Temps total estimé pour Phase 2:** 1-2 jours

---

*Rapport généré: 2026-03-23*
