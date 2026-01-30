📌 Résumé de l'avancement du projet
👥 Équipe

Kameni Agnès — Responsable Données

Abdou Sylla — Responsable Modèles

1. Préparation du jeu de données (Kameni Agnès)

Depuis le 23 novembre, Agnès a :

Mis en place l'environnement Python et la structure du projet.

Développé un script pour télécharger un sous-ensemble du dataset LAION (≈ 30 000 exemples, ~19 810 valides).

Réalisé une analyse exploratoire complète : histogrammes, visualisation d'images, détection d'anomalies.

Implémenté un pipeline de filtrage de qualité brute incluant :

filtre NSFW (punsafe > 0.3 ),

filigrane de filtre (pwatermark > 0.2 ),

suppression des images trop petites (< 256 px),

Nettoyage des légendes (3 à 40 mots),

seuil esthétique minimal (≥ 6,5).

Généré un rapport statistique détaillé et obtenu un dataset propre ( laion_aesthetic_light.csv ).

📌 Conclusion : la totalité des tâches prévues en Semaine 1 et Semaine 2 côté Data sont terminées.

2. Évaluation des modèles (Abdou Sylla)

Depuis le 23 novembre, Abdou a :

Installez OpenCLIP et mettez en place un pipeline d'évaluation reproductible.

Évalué le modèle OpenCLIP ViT-B/32 en zero-shot sur CIFAR-10.

Résultat : 93.6% d'accuracy , serviteur de baseline.

Réalisé un réglage expérimental sur CIFAR-10 (15 époques).

Résultat : baisse à ~56% due à un oubli catastrophique — comportement attendu.

Comparé performances RAW vs CLEAN :

BRUT : 89,0%

CLEAN : 89.8%
→ Les données filtrées donnent un modèle plus performant .

Évalué la robustesse sur ImageNetV2 :

Le modèle CLEAN reste légèrement meilleur que RAW.

📌 Conclusion : Abdou a complété la Semaine 1, avancé sur la Semaine 2, et commencé des tests normalement prévus pour plus tard.

3. Synthèse

Le pipeline Data est propre, fonctionnel et validé .

Les premières expériences montrent que la qualité des données améliore réellement le modèle .

L'équipe dispose désormais :

d'un jeu de données propre,

d'une baseline solide,

d'un pipeline d'évaluation robuste.

🎯 Prochaine étape : calcul du CLIPScore et construction des datasets « light » et « strong ».