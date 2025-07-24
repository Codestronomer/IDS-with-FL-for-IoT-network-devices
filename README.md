# 🚨 Federated Anomaly Detection in IIoT using XGBoost

A federated learning system that leverages client-local training of XGBoost models to detect cyber-attacks in Industrial Internet of Things (IIoT) networks without sharing raw data.

---

## 📌 Overview

This project aims to build a privacy-preserving **federated anomaly detection** system for IIoT environments. Traditional centralized models pose data privacy concerns. Here, **federated learning** is applied using **Flower**, allowing each client to train locally using an **XGBoost classifier**, while model evaluation and metrics are aggregated at the server level.

### Goals:
- Detect various IIoT cyber-attacks using a distributed dataset.
- Evaluate communication cost, training time, and model accuracy per client.
- Address class imbalance and simulate a real-world federated setting.

---

## 📂 Project Structure
```
📁 project_root/
│
├── 📄 main.py                  # Entry point to run the federated simulation (Flower NumPyClient definition using XGBoost Custom Flower server with non-aggregated strategy)
├── 📄 config.py                # Hyperparameters and constants
├── 📁 notebooks/
│   ├── preprocess.ipynb      # Data cleaning, encoding, and merging.
|   └── IDS with FL for edge devices.ipynb # Visualization, Feature selection, Model training, FL simulations and experiments
├── 📁 datasets/
│   ├── merged_edge_iiotset.csv        # Preprocessed dataset
|   └── edge_iiotset.parquet
├── 📁 outputs/
│   ├── figure_4_5_accuracy.png
│   ├── figure_4_6_communication.png
│   └── figure_4_7_training_time.png
├── 📄 requirements.txt         # Python dependencies
└── README.md                   # Project documentation
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Codestronomer/IDS-with-FL-for-IoT-network-devices.git
cd IDS-with-FL-for-IoT-network-devices
```

### 2. Create Python Environment
```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
Make sure you have:
-	Python 3.10 or higher
- ray for parallel execution
- xgboost, scikit-learn, pandas, matplotlib, flwr

### 4. How to run 🚀
Federated learning simulation
```bash
python main.py
```

## 📊 Data Overview

- **Dataset**: IIoT Dataset containing labeled normal and attack traffic.  
- **Attributes**: 63 columns including timestamps, protocols, payloads, and flags.

### 🔄 Preprocessing Steps

- Dropped irrelevant or incomplete features  
- Handled missing values and infinite values  
- Encoded categorical variables (e.g., HTTP method, MQTT types)  
- Normalized features using `StandardScaler`  
- Balanced label distributions across clients

---
# 📈 Results

## 📌 Evaluation Metrics
- **Accuracy (per round):** >99.7%
- **Loss:** ~0.011 consistently  
- **Training Time per Client:** See *Figure 4.7*
- **Communication Overhead:** See *Figure 4.6*

## 🖼️ Visualizations
- `figure_4_/Users/macbookpro/Documents/dev/intrusion_detection_system/figure_4_5_accuracy.png5_accuracy.png`: Accuracy of each client  
/Users/macbookpro/Documents/dev/intrusion_detection_system/figure_4_5_accuracy.png
- `figure_4_6_communication.png`: Bytes exchanged by each client  
- `figure_4_7_training_time.png`: Local training duration in seconds  

---

## 🧠 Model Architecture
- **Model:** XGBoost (`XGBClassifier`)
- **Objective:** `binary:logistic`
- **Tree Method:** `hist`
- **Custom Parameters:**
  - `max_depth=5`
  - `learning_rate=0.2`
  - `gamma=1`
  - `scale_pos_weight` is dynamically set per client to handle class imbalance


### 📊 Distributed Training History

| **Round** | **Loss**       | **Accuracy**         |
|-----------|----------------|----------------------|
| 1         | 0.0110927050   | 0.9986692858500729   |
| 2         | 0.0116520783   | 0.9982256020278834   |
| 3         | 0.0119891096   | 0.9982889733840304   |
| 4         | 0.0119845638   | 0.9980989797858184   |
| 5         | 0.0123107134   | 0.9981622306717364   |
| 6         | 0.0127927748   | 0.9979088777644002   |
| 7         | 0.0126462125   | 0.9977186311787072   |
| 8         | 0.0137699810   | 0.9975286737215640   |
| 9         | 0.0127546837   | 0.9982257144667638   |
| 10        | 0.0111649670   | 0.9983523447401774   |

# 📦 Deployment

**Not deployed in production.** For API-based deployment:

- Serialize trained XGBoost models with `.save_model()`
- Wrap in a FastAPI or Flask service
- Expose `/predict` and `/health` endpoints

---

# 💡 Tips / Gotchas

- Ensure `ray.init()` does **not enable dashboard** on Windows to avoid errors
- Always **normalize features** for XGBoost consistency
- **Stratified data split** is critical to retain label balance across clients
- Check **client logs** carefully for anomalies or dropped evaluations
- Avoid **over-tuning XGBoost params** on highly imbalanced datasets; use `scale_pos_weight`

---

# 🤝 Contributing

We welcome community contributions!

1. Fork the repository  
2. Create a feature branch:  
   ```bash
   git checkout -b feature/your-feature-name
   ```
3.  Commit Changes
   ```bash
   git commit -m "Add new feature"
   ```
4.  Push to Github
    ```bash
    git push origin feature/your-feature-name
    ```
5.	Open a Pull Request
For bug reports or feature suggestions, please open an issue.

## 📄 License

This project is licensed under the MIT License.
See LICENSE file for details.

## References
 - [Flower Documentation](https://flower.ai/docs/)
 - [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)
 - [Ray for Distributed Python](https://xgboost.readthedocs.io/en/stable/)
 - Dataset Paper: [Edge-IIoTset Cyber Security Dataset of IoT & IIoT](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot/data?status=pending&suggestionBundleId=483)

## 📆 Project Timeline: Federated Learning Intrusion Detection system with Edge-IIoTset

### 🗂 Dataset Used
- **Name:** Edge-IIoTset
- **Description:** A comprehensive dataset for evaluating intrusion detection in Industrial IoT systems, containing benign and malicious traffic records across multiple attack types.

---

### 📅 Timeline Breakdown

| **Week**   | **Activities**                                                                                                                                             | **Deliverables**                                                                                   |
|------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| **Week 1** | **Project Setup & Data Preprocessing**<br>- Define architecture, tools, and project scope (Flower, XGBoost)<br>- Load and merge Edge-IIoTset dataset<br>- Clean NaNs/infs, encode categorical features, normalize data<br>- Partition dataset into client-local shards | - Cleaned & preprocessed Edge-IIoTset CSV<br>- Data partitioning script<br>- Data distribution report |
| **Week 2** | **Centralized Model Training**<br>- Train XGBoost model on the full aggregated dataset<br>- Tune hyperparameters using RandomizedSearchCV<br>- Evaluate accuracy, F1-score, confusion matrix<br>- Save optimal model | - Trained centralized XGBoost model (`.pkl`)<br>- Evaluation report<br>- Confusion matrix & metrics chart |
| **Week 3** | **Federated Framework Setup**<br>- Install and configure Flower with simulation support<br>- Implement `XGBoostClient` class (fit, evaluate)<br>- Define `FederatedXGBoostStrategy` with custom aggregation<br>- Simulate 10 clients training locally | - Flower-based federated training script<br>- Custom strategy and client logic<br>- Initial simulation logs |
| **Week 4** | **Model Evaluation & Comparison**<br>- Evaluate distributed model accuracy over 10 rounds<br>- Generate loss and accuracy tables<br>- Compare centralized vs federated performance<br>- Analyze local data limitations | - Performance comparison table<br>- Markdown tables for round-wise metrics<br>- Written discussion (e.g., 4.5.1 section) |
| **Week 5** | **Imbalance Handling & Optimization**<br>- Handle class imbalance using `scale_pos_weight` or local SMOTE<br>- Refine simulation strategy (e.g., fraction_fit, num_rounds)<br>- Log precision, recall, F1 across rounds | - Updated federated training with imbalance handling<br>- Per-client performance metrics<br>- Tuned strategy settings |
| **Week 6** | **Documentation & Finalization**<br>- Complete Chapter 4 and 5 sections including 4.5.1 and 5.3<br>- Finalize tables, charts, and figures<br>- Write abstract, conclusion<br>- Prepare presentation or defense materials | - Final project report (PDF)<br>- Polished codebase with README<br>- Slide deck |

---
