import os
import gzip
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Resolve paths
script_dir = os.path.dirname(os.path.abspath(__file__))
vcf_file = os.path.join(script_dir, "output.vcf.gz")

# Parse VCF
variants = []
with gzip.open(vcf_file, "rt") as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.strip().split("\t")
        variants.append(parts)

# Create DataFrame
columns = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", "SAMPLE"]
df = pd.DataFrame(variants, columns=columns)

# 1. Variant Type Distribution
variant_type = []
for i in range(len(df)):
    ref = df.loc[i, "REF"]
    alt = df.loc[i, "ALT"]
    if len(ref) == 1 and len(alt) == 1:
        variant_type.append("SNP")
    else:
        variant_type.append("INDEL")

counts = {
    "SNP": variant_type.count("SNP"),
    "INDEL": variant_type.count("INDEL")
}

plt.figure(figsize=(5, 5))
plt.pie(counts.values(),
        labels=counts.keys(),
        autopct="%1.1f%%",
        startangle=90,
        colors=["#3498db", "#e74c3c"])
plt.title("Variant Type Distribution")
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "variant_type_distribution.png"), dpi=150)
plt.close()

# 2. Variant Quality Scores Histogram
df["QUAL"] = df["QUAL"].astype(float)
plt.figure(figsize=(7, 4))
plt.hist(df["QUAL"], bins=20, color="#2ecc71", edgecolor="#27ae60", alpha=0.8)
plt.xlabel("Quality Score (QUAL)")
plt.ylabel("Number of Variants")
plt.title("Distribution of Variant Quality Scores")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "variant_quality_scores.png"), dpi=150)
plt.close()

# 3. Variant Quality Across the Genome
plt.figure(figsize=(10, 4))
plt.plot(df["POS"].astype(int), df["QUAL"], "o", markersize=4, color="#9b59b6", alpha=0.7)
plt.xlabel("Genomic Position (Chromosome 20)")
plt.ylabel("Quality Score (QUAL)")
plt.title("Variant Quality Across Genomic Positions")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "variant_quality_scatter.png"), dpi=150)
plt.close()

print("DeepVariant Figures successfully generated!")
