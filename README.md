# Fine-Tuning TATR for School Schedules Extraction from Photos

## Overview
In this document, we will cover the fine-tuning process of TATR (Text-Aware Transformer for Retrieval) for extracting school schedules from photos. This guide includes setup instructions, hardware specifications, dataset information, and links to relevant GitHub issues.

## Setup Guide
1. **Clone the Repository**  
   ```bash
   git clone https://github.com/InspireEureka/table-transformer-orari-docenti.git
   cd table-transformer-orari-docenti
   ```  

2. **Install Dependencies**  
   Make sure to have Python 3.8+ installed along with pip, then run:
   ```bash
   pip install -r requirements.txt
   ```  

3. **Configure Environment Variables**  
   Set up any necessary environment variables as per the configuration file provided in the repository.

## Hardware Specifications
- **GPU:** RTX 3060
- **RAM:** 32GB

## Dataset Information
The dataset used for fine-tuning includes:
- **100 personal photos**  
  These photos are captured in various lighting conditions and angles for robustness.
- **2000 photos from colleagues**  
  This larger dataset from multiple sources helps in generalizing the model.

## Fine-Tuning Process
1. **Data Preparation**  
   Preprocess the data before feeding it into the model.

2. **Model Training**  
   Use the following command to start fine-tuning:
   ```bash
   python train.py --data_path <path_to_dataset> --output_dir <model_output_directory>
   ```

3. **Evaluation**  
   Evaluate the model on a separate validation set to check its performance.

## Relevant GitHub Issues
- [Issue #1](https://github.com/InspireEureka/table-transformer-orari-docenti/issues/1)  
- [Issue #2](https://github.com/InspireEureka/table-transformer-orari-docenti/issues/2)  

Please refer to these links for discussions around the setup and improvement suggestions.