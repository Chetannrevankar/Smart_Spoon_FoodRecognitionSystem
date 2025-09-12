<p align="center">
  <h1 align="center">🍝🍚🥗🍞 Smart Spoon Food Recognition System 🍝🍚🥗🍞</h1>
</p>

## 🎯 Introduction and Objective
The Smart Spoon Food Recognition System is an AI-powered food analytics platform built to help users, especially health conscious individuals and those with dietary restrictions, monitor and optimize their food intake. Leveraging deep learning for food recognition, it provides instant, easy to understand feedback about food types and estimated salt content from a photo, along with advanced data logging and sentiment analysis. The project aims to assist users particularly gym goers, athletes, and patients managing salt intake for both dietary tracking and medical guidance.
***
## ✨ Features
- **AI-Based Food Recognition:** Identifies food type from images using a MobileNetV2-based CNN model.
- **Salt Content Estimation:** Estimates salt content based on the recognized food type.
- **Usage Logging:** Logs every prediction, confidence, and salt content for analysis.
- **Market and Sentiment Analysis:** Processes usage, survey, and feedback data for insights.
- **User Feedback & Sentiment:** Collects textual feedback and classifies user sentiment (Positive/Neutral/Negative).
- **Interactive Dashboard:** A user friendly Streamlit app interface for easy use and live visualization.
- **Comprehensive Data Analysis:** Includes tools for survey insights, behavioral prediction, and sentiment breakdown.
***
##  Workflow Diagram
*This diagram shows the complete data flow and architecture of the system.*
![Flow Diagram](https://github.com/Chetannrevankar/Smart_Spoon_FoodRecognitionSystem/raw/main/flow_diagram.png)
***
## 🛠️ Tech Stack & Key Libraries
| Technology/Library | Purpose/Why Used |
|---|---|
| **Python** | Core development and scripting language. |
| **TensorFlow / Keras** | Deep learning for training and deploying food recognition CNNs using transfer learning. |
| **MobileNetV2** | Lightweight CNN pre-trained on ImageNet, fine-tuned for high-accuracy food classification with modest resource requirements. |
| **Pandas** | Tabular data processing, aggregation, and CSV/Excel handling. |
| **scikit-learn** | Machine learning for behavioral prediction and model evaluation. |
| **NLTK** | NLP and sentiment analysis for parsing, scoring, and classifying textual user feedback. |
| **Matplotlib / Seaborn** | Data visualization for all analytics, insights, and evaluation charts. |
| **Streamlit** | Rapid, interactive web app and dashboard deployment. |
| **OpenCV, PIL** | Food image loading and pre-processing. |
| **Ultralytics, PyTorch** | Included for possible expansions like advanced object detection or future research. |
***
## ⚙️ How to Use
1.  **Clone the repository:**
    ```
    git clone https://github.com/Chetannrevankar/Smart_Spoon_FoodRecognitionSystem.git
    cd Smart_Spoon_FoodRecognitionSystem
    ```
2.  **Install Dependencies:**
    ```
    pip install -r requirements.txt
    ```
3.  **Prepare Data:**
    - Place food images inside `data/food_images/` (by category folders).
    - Ensure CSV files (`usage_logs.csv`, `user_feedback.csv`, etc.) are in place or will be auto-created.
4.  **Train the Model (Optional):**
    ```
    python train_food_model.py
    ```
    *(The repository includes a pretrained model for quick use.)*
5.  **Launch the Application:**
    ```
    streamlit run streamlit_app.py
    ```
6.  **Interact with the App:**
    - Upload food images for instant recognition and salt estimation.
    - View usage logs and feedback analysis.
    - Submit and view sentiment for your own feedback.
***
## 🌟 Why to Use
-   **💪 Health & Fitness:** A much-needed tool for gym-goers, dieters, athletes, and nutrition-conscious individuals to track salt intake and food types easily.
-   **❤️ Medical Perspective:** Vital for patients with hypertension, kidney issues, or cardiac conditions who must monitor and limit salt consumption.
-   **🔬 Research & Insights:** Enables professionals to analyze user satisfaction, food choice trends, and overall behavioral patterns from survey data.
-   **👍 Ease of Use:** A streamlined, no-hassle interface requiring only image uploads and brief feedback text.
***
## 👥 Target Users
- Gym enthusiasts, bodybuilders, and athletes.
- Patients or users advised for salt intake control (e.g., hypertension, cardiac, kidney care).
- Doctors/nutritionists seeking digital tools for patient advice.
- Data scientists and researchers in food tech and health informatics domains.
***
## 🎬 Sample Outputs
- 📺[Project Frontend Demo(Streamlit)](https://github.com/Chetannrevankar/Smart_Spoon_FoodRecognitionSystem/blob/main/outputs/project_frontend.mp4)
- 🧑‍💻[Project Backend Demo](https://github.com/Chetannrevankar/Smart_Spoon_FoodRecognitionSystem/blob/main/outputs/project_backend.mp4)
***
## ✍️ Authors
-   [**Chetan N Revankar**](https://github.com/Chetannrevankar)
    
-   [**Syed Allahadad Hassan**](https://github.com/SyedHassan007)
***
## 📄 License
This project is licensed under the MIT License. See the `LICENSE` file for details.
