# Notebooks — exploration initiale

Les deux premiers pas du projet, datés de février 2026 : décodage de fichiers
`.fit` avec le SDK Garmin, puis construction des features de séance
(découplage cardiaque, masque de course, efficience).

Ils sont conservés parce qu'ils montrent d'où part le projet. **Ils ne sont pas
exécutables tels quels** : ils lisent deux fichiers `.fit` d'exemple qui ne sont
pas versionnés, `data/` étant ignoré par git et surveillé par une fixture de
test. Leurs imports, en revanche, résolvent toujours contre le code actuel.

Les sorties ont été retirées. Elles pesaient 1,2 Mo de figures Plotly, que
l'aperçu de notebook de GitHub ne rend pas — ce poids n'achetait rien.

Ce que ces notebooks faisaient à la main vit maintenant dans
`src/layer_a_signal/` et se lance par `python -m src.cli ingest`.
