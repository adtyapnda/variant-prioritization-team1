# DeepVariant Methodology

## Objective
To replicate the DeepVariant paper by setting up the official codebase and running the model on the provided example dataset.

## Repository
[DeepVariant GitHub Repository](https://github.com/google/deepvariant)

## Environment
- Operating System: Windows
- Terminal: PowerShell
- Runtime: Docker (CPU)

## Steps Followed

### 1. Clone Repository
Cloned the official DeepVariant GitHub repository.

### 2. Setup
Installed Docker and pulled the official DeepVariant Docker image.

### 3. Download Test Data
Downloaded the official quickstart test dataset provided by the DeepVariant repository.

### 4. Run DeepVariant
Executed the DeepVariant quickstart pipeline on the example BAM file using the reference genome.

### 5. Output Generated
The pipeline completed successfully and generated:

- output.vcf.gz
- output.vcf.gz.tbi

The log reported:

- Total variants written: 288

## Output Description

### output.vcf.gz
Contains the predicted SNPs and small insertion/deletion variants detected by DeepVariant.

### output.vcf.gz.tbi
Index file for the VCF, enabling fast access.

## Status

✔ Repository setup completed.

✔ Example dataset executed successfully.

✔ Output files generated and uploaded to the repository.

## Future Work

- Run DeepVariant on additional datasets.
- Analyze VCF variants in detail.
- Compare results with other variant calling methods.
