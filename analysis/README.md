# Enrichment Analysis
Predicted drug responses can be further analyzed through Gene Ontology (GO) and Kyoto Encyclopedia of Genes and Genomes (KEGG) enrichment analyses. We created a .csv file containing the top 20% lowest IC50 predictions for drug-cell line pairings made by our model. We did this to find out whether the genes and biological pathways associated with our model's predictions correlate with already established cancer-related pathways. The workflow was:

- Select highly ranked predicted drug-cell line responses and save them as top20_predicted_drug_responses.csv.
- Collate the predicted drugs and use the ChEMBL database to map drug names to ChEMBL IDs and identify their associated biological targets, saving the results as drug_ids.csv.
- Perform GO Biological Process and KEGG pathway enrichment analysis on the identified genes.

# References
- Mendez, D. et al. (2019). ChEMBL: Towards direct deposition of bioassay data. Nucleic Acids Research, 47(D1), D930–D940. https://doi.org/10.1093/nar/gky1075
