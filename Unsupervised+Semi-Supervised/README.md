# Unsupervised + Semi-Supervised Learning

This folder contains comprehensive projects demonstrating unsupervised and semi-supervised learning techniques. These projects cover clustering, dimensionality reduction, anomaly detection, recommendation systems, and advanced neural network architectures like autoencoders and transformer-based models.

## 🔍 Projects Overview

### 1. [Autoencoder for Noise Reduction](./Autoencoder_for_Noise_Reducon.ipynb)
**Goal**: Build an autoencoder to reconstruct clean images from noisy ones using the MNIST dataset.

**Key Features**:
- Dense and Convolutional autoencoder architectures
- Noise addition and denoising capabilities
- MNIST dataset processing and visualization
- Performance comparison between architectures
- Dropout regularization analysis

**Technologies**: TensorFlow/Keras, Autoencoders, CNN, MNIST dataset

**Architectures**:
- **Dense Autoencoder**: 784 → 128 → 64 → 128 → 784
- **Convolutional Autoencoder**: Conv2D + MaxPooling + UpSampling
- **With Dropout**: Regularization for better generalization

**Performance**: Convolutional autoencoder outperforms dense version for image denoising

### 2. [User-Based Recommender System](./Build_a_User_Based_Recommender_System.ipynb)
**Goal**: Recommend items to users based on preferences of similar users using User-Based Collaborative Filtering.

**Key Features**:
- MovieLens dataset integration
- User similarity computation (cosine similarity)
- Collaborative filtering algorithms
- Recommendation generation and evaluation
- Performance metrics (RMSE, MAE)

**Technologies**: pandas, NumPy, scikit-learn, MovieLens dataset

### 3. [Dimensionality Reduction with t-SNE](./Dimensionality_Reducon_with_t_SNE.ipynb)
**Goal**: Reduce high-dimensional data to 2D/3D for visualization and analysis using t-SNE.

**Key Features**:
- t-SNE implementation and parameter tuning
- High-dimensional data visualization
- Cluster analysis and interpretation
- Comparison with other dimensionality reduction techniques

**Technologies**: scikit-learn, matplotlib, t-SNE, PCA

### 4. [Email Spam Classifier](./Email_Spam_Classifier.ipynb)
**Goal**: Classify emails as spam or legitimate using machine learning techniques.

**Key Features**:
- Text preprocessing and feature extraction
- Multiple classification algorithms
- Performance evaluation and comparison
- Feature importance analysis

**Technologies**: scikit-learn, NLTK, TF-IDF, Classification algorithms

### 5. [Gaussian Mixture Models for Speaker Identification](./Gaussian_Mixture_Models_for_Speaker_Idenfication.ipynb)
**Goal**: Identify speakers from audio data using Gaussian Mixture Models.

**Key Features**:
- Audio feature extraction (MFCC)
- GMM clustering for speaker identification
- Model training and evaluation
- Speaker verification capabilities

**Technologies**: scikit-learn, librosa, GMM, Audio processing

### 6. [Hierarchical Clustering on E-commerce Data](./Hierarchical_Clustering_on_E_commerce_Data.ipynb)
**Goal**: Perform hierarchical clustering on e-commerce customer data to identify customer segments.

**Key Features**:
- Customer segmentation analysis
- Hierarchical clustering algorithms
- Dendrogram visualization
- Cluster interpretation and business insights

**Technologies**: scikit-learn, pandas, matplotlib, Hierarchical clustering

### 7. [Outlier Detection in Financial Data](./Outlier_Detection_in_Financial_Data.ipynb)
**Goal**: Detect anomalies and outliers in financial datasets using various detection methods.

**Key Features**:
- Multiple outlier detection algorithms
- Financial data preprocessing
- Anomaly visualization and analysis
- Risk assessment applications

**Technologies**: scikit-learn, pandas, Isolation Forest, One-Class SVM

### 8. [Question Answering System (Using Transformers)](./Question_Answering_System_(Using_Transformers).ipynb)
**Goal**: Build a question-answering system using transformer models for document-based Q&A.

**Key Features**:
- Transformer model integration
- Document processing and indexing
- Question-answer matching
- Context-aware response generation

**Technologies**: Transformers, BERT, Question Answering, NLP

### 9. [Self-Supervised Pretraining on Images](./Self_Supervised_Pretraining_on_Images.ipynb)
**Goal**: Implement self-supervised learning techniques for image representation learning.

**Key Features**:
- Contrastive learning approaches
- Image augmentation strategies
- Representation learning without labels
- Transfer learning applications

**Technologies**: PyTorch, Contrastive learning, Data augmentation

### 10. [Semi-Supervised Learning for Document Labeling](./Semi_Supervised_Learning_for_Document_Labeling.ipynb)
**Goal**: Use semi-supervised learning to classify documents with limited labeled data.

**Key Features**:
- Label propagation algorithms
- Co-training approaches
- Active learning strategies
- Performance with limited labels

**Technologies**: scikit-learn, Semi-supervised learning, Document classification

### 11. [Sentiment Analysis with Fine-Tuned BERT](./Sentiment_Analysis_with_Fine_Tuned_BERT.ipynb)
**Goal**: Perform sentiment analysis using fine-tuned BERT models for text classification.

**Key Features**:
- BERT model fine-tuning
- Sentiment classification (positive/negative/neutral)
- Transfer learning from pre-trained models
- Performance evaluation and comparison

**Technologies**: Transformers, BERT, Fine-tuning, Sentiment analysis

### 12. [Text Summarizer with T5](./Text_Summarizer_with_T5.ipynb)
**Goal**: Generate text summaries using T5 (Text-to-Text Transfer Transformer) models.

**Key Features**:
- T5 model implementation
- Abstractive text summarization
- Multiple summary lengths
- Quality evaluation metrics

**Technologies**: Transformers, T5, Text summarization, NLP

### 13. [Zero-Shot Text Classification](./Zero_Shot_Text_Classification.ipynb)
**Goal**: Classify text into categories without training examples using zero-shot learning.

**Key Features**:
- Zero-shot classification with pre-trained models
- Natural language inference
- Category description-based classification
- Performance without labeled training data

**Technologies**: Transformers, Zero-shot learning, Text classification

## 🛠️ Common Technologies Used

- **Deep Learning**: TensorFlow, Keras, PyTorch, Autoencoders
- **ML Libraries**: scikit-learn, pandas, NumPy
- **NLP**: Transformers, BERT, T5, NLTK
- **Visualization**: matplotlib, seaborn, plotly
- **Audio Processing**: librosa, MFCC features
- **Clustering**: K-means, Hierarchical, GMM, DBSCAN

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow torch transformers scikit-learn
pip install pandas numpy matplotlib seaborn
pip install nltk librosa  # For NLP and audio processing
pip install plotly  # For advanced visualizations
```

### Running the Projects

1. **Choose a project** from the list above
2. **Open the notebook** in Jupyter or Google Colab
3. **Install dependencies** as specified in each notebook
4. **Download datasets** (MovieLens, audio data, text corpora)
5. **Follow the step-by-step implementation**
6. **Experiment** with different algorithms and parameters

### Google Colab Integration
Most notebooks include direct links to run in Google Colab:
- Click the "Open In Colab" badge at the top of each notebook
- Enable GPU for transformer models and deep learning
- Some projects may require additional setup for audio processing

## 📊 Key Concepts Covered

### Unsupervised Learning
- **Clustering**: K-means, Hierarchical, GMM, DBSCAN
- **Dimensionality Reduction**: PCA, t-SNE, UMAP
- **Anomaly Detection**: Isolation Forest, One-Class SVM
- **Autoencoders**: Dense and Convolutional architectures
- **Recommendation Systems**: Collaborative filtering

### Semi-Supervised Learning
- **Label Propagation**: Graph-based semi-supervised learning
- **Co-training**: Multi-view learning approaches
- **Self-training**: Iterative learning with pseudo-labels
- **Active Learning**: Intelligent sample selection

### Advanced Techniques
- **Self-Supervised Learning**: Learning without explicit labels
- **Contrastive Learning**: Learning representations through comparison
- **Zero-Shot Learning**: Classification without training examples
- **Transfer Learning**: Leveraging pre-trained models

## 🎯 Learning Objectives

After completing these projects, you will understand:

- **Clustering Algorithms**: Different approaches to grouping data
- **Dimensionality Reduction**: Techniques for data visualization and compression
- **Anomaly Detection**: Identifying outliers and unusual patterns
- **Autoencoders**: Neural networks for data reconstruction and generation
- **Recommendation Systems**: Collaborative and content-based filtering
- **Semi-Supervised Learning**: Learning with limited labeled data
- **Transformer Models**: BERT, T5, and other pre-trained language models

## 🔧 Model Architectures

### Autoencoder Architecture
```
Input Layer (784 for MNIST)
↓
Encoder: Dense(128) → Dense(64)
↓
Bottleneck (Latent Space)
↓
Decoder: Dense(128) → Dense(784)
↓
Output Layer (Reconstruction)
```

### Convolutional Autoencoder
```
Input: (28, 28, 1)
↓
Encoder: Conv2D(32) → MaxPool → Conv2D(16) → MaxPool
↓
Bottleneck: (7, 7, 16)
↓
Decoder: Conv2D(16) → UpSample → Conv2D(32) → UpSample → Conv2D(1)
↓
Output: (28, 28, 1)
```

## 📈 Performance Optimization

Each project includes techniques for:
- **Hyperparameter Tuning**: Optimizing algorithm parameters
- **Model Selection**: Choosing appropriate algorithms for specific tasks
- **Feature Engineering**: Creating meaningful features from raw data
- **Evaluation Metrics**: Measuring performance without ground truth
- **Visualization**: Understanding results through plots and graphs

## 🎨 Applications

These techniques can be applied to:

- **Recommendation Systems**: E-commerce, streaming platforms
- **Anomaly Detection**: Fraud detection, network security
- **Customer Segmentation**: Marketing, personalization
- **Data Compression**: Image compression, feature reduction
- **Natural Language Processing**: Text classification, summarization
- **Computer Vision**: Image denoising, representation learning
- **Audio Processing**: Speaker identification, music analysis

## 📚 Datasets Used

- **MNIST**: Handwritten digits for autoencoder training
- **MovieLens**: Movie ratings for recommendation systems
- **Text Corpora**: Various text datasets for NLP tasks
- **Audio Data**: Speech samples for speaker identification
- **Financial Data**: Market data for anomaly detection
- **E-commerce Data**: Customer behavior for clustering

## 🤝 Contributing

Feel free to:
- Add new unsupervised/semi-supervised projects
- Improve existing implementations
- Share performance optimizations
- Report issues or suggest enhancements
- Contribute new datasets or use cases

## 📖 Additional Resources

- [scikit-learn Unsupervised Learning](https://scikit-learn.org/stable/unsupervised_learning.html) - Official documentation
- [Autoencoder Tutorial](https://blog.keras.io/building-autoencoders-in-keras.html) - Keras guide
- [t-SNE Paper](https://lvdmaaten.github.io/tsne/) - Original research
- [BERT Paper](https://arxiv.org/abs/1810.04805) - Transformer research
- [Self-Supervised Learning Survey](https://arxiv.org/abs/1902.06162) - Comprehensive review

## 🔍 Troubleshooting

Common issues and solutions:
- **Memory Issues**: Use batch processing for large datasets
- **Convergence**: Adjust learning rates and initialization
- **Clustering Quality**: Try different algorithms and parameters
- **Model Performance**: Experiment with different architectures

## 🚀 Advanced Topics

For those looking to extend these projects:
- **Deep Clustering**: Neural network-based clustering
- **Generative Models**: GANs and VAEs for data generation
- **Multi-modal Learning**: Combining different data types
- **Federated Learning**: Distributed unsupervised learning

---

**Note**: Start with clustering and dimensionality reduction projects to understand basic unsupervised learning concepts, then progress to more advanced techniques like autoencoders and transformer models. Each project demonstrates different approaches to learning without explicit supervision.
