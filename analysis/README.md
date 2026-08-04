# Enrichment Analysis
Predicted drug responses can be further analyzed through Gene Ontology (GO) and Kyoto Encyclopedia of Genes and Genomes (KEGG) enrichment analyses. We created a .csv file containing the top 20% lowest IC50 predictions made by our model. We did this to find out whether the genes and biological pahtways associated with our model's predictions correlate with already established cancer-related pathways. The workflow was:

- Select highly ranked predicted drug cell line responses and save it as top20_predicted_drug_reponses.csv
- Collated a drug list from reponses and identify biological targets of these drugs and saved it as drug_ids.csv
- Collated a Perform GO enrichment analysis. Identify enriched biological pathways.
