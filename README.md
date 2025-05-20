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


# 📆 6-Week Project Timeline: Federated XGBoost-Based Intrusion Detection with Edge-IIoTset

## 🗂 Dataset Used
- **Name:** Edge-IIoTset
- **Description:** A comprehensive dataset for evaluating intrusion detection in Industrial IoT systems, containing benign and malicious traffic records across multiple attack types.

---

## 📅 Timeline Breakdown

| **Week**   | **Activities**                                                                                                                                             | **Deliverables**                                                                                   |
|------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| **Week 1** | **Project Setup & Data Preprocessing**<br>- Define architecture, tools, and project scope (Flower, XGBoost)<br>- Load and merge Edge-IIoTset dataset<br>- Clean NaNs/infs, encode categorical features, normalize data<br>- Partition dataset into client-local shards | - Cleaned & preprocessed Edge-IIoTset CSV<br>- Data partitioning script<br>- Data distribution report |
| **Week 2** | **Centralized Model Training**<br>- Train XGBoost model on the full aggregated dataset<br>- Tune hyperparameters using RandomizedSearchCV<br>- Evaluate accuracy, F1-score, confusion matrix<br>- Save optimal model | - Trained centralized XGBoost model (`.pkl`)<br>- Evaluation report<br>- Confusion matrix & metrics chart |
| **Week 3** | **Federated Framework Setup**<br>- Install and configure Flower with simulation support<br>- Implement `XGBoostClient` class (fit, evaluate)<br>- Define `FederatedXGBoostStrategy` with custom aggregation<br>- Simulate 10 clients training locally | - Flower-based federated training script<br>- Custom strategy and client logic<br>- Initial simulation logs |
| **Week 4** | **Model Evaluation & Comparison**<br>- Evaluate distributed model accuracy over 10 rounds<br>- Generate loss and accuracy tables<br>- Compare centralized vs federated performance<br>- Analyze local data limitations | - Performance comparison table<br>- Markdown tables for round-wise metrics<br>- Written discussion (e.g., 4.5.1 section) |
| **Week 5** | **Imbalance Handling & Optimization**<br>- Handle class imbalance using `scale_pos_weight` or local SMOTE<br>- Refine simulation strategy (e.g., fraction_fit, num_rounds)<br>- Log precision, recall, F1 across rounds | - Updated federated training with imbalance handling<br>- Per-client performance metrics<br>- Tuned strategy settings |
| **Week 6** | **Documentation & Finalization**<br>- Complete Chapter 4 and 5 sections including 4.5.1 and 5.3<br>- Finalize tables, charts, and figures<br>- Write abstract, conclusion<br>- Prepare presentation or defense materials | - Final project report (PDF)<br>- Polished codebase with README<br>- Slide deck |

---
