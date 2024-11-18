# food_classifier/settings.py
import os

INSTALLED_APPS = [
    # ... other apps ...
    'food_classifier',
    'rest_framework',
    'corsheaders',
]

MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
MEDIA_URL = '/media/'

# food_classifier/models.py
from django.db import models
import json
from datetime import datetime

class FoodImage(models.Model):
    image = models.ImageField(upload_to='food_images/%Y/%m/%d/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    predictions = models.JSONField(null=True)
    
    def get_predictions(self):
        return json.loads(self.predictions) if self.predictions else {}

# food_classifier/ml_models.py
import torch
import torchvision.transforms as transforms
from PIL import Image
import mlflow
import os
from .unified_food_classifier import UnifiedFoodClassifier, FOOD_CATEGORIES

class ModelService:
    def __init__(self):
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        self.load_model()

    def load_model(self):
        mlflow.set_tracking_uri("http://localhost:5000")
        latest_run = mlflow.search_runs(
            experiment_ids=['1'],
            order_by=['start_time DESC']
        ).iloc[0]
        
        model_path = latest_run.artifact_uri + "/model"
        self.model = mlflow.pytorch.load_model(model_path)
        self.model.eval()

    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        
        with torch.no_grad():
            predictions = self.model(image_tensor)
            
        return self.process_predictions(predictions)

    def process_predictions(self, predictions):
        sandwich_probs = torch.softmax(predictions['sandwich_type'][0], dim=0)
        ingredient_probs = torch.sigmoid(predictions['ingredients'][0])
        dressing_probs = torch.softmax(predictions['dressing_rec'][0], dim=0)
        
        return {
            'sandwich_type': {
                k: float(sandwich_probs[v]) 
                for k, v in FOOD_CATEGORIES['SANDWICH_TYPES'].items()
            },
            'ingredients': {
                k: float(ingredient_probs[v])
                for k, v in FOOD_CATEGORIES['SALAD_INGREDIENTS'].items()
                if float(ingredient_probs[v]) > 0.5
            },
            'recommended_dressings': {
                k: float(dressing_probs[v])
                for k, v in FOOD_CATEGORIES['DRESSING_TYPES'].items()
            }
        }

# food_classifier/views.py
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from django.conf import settings
from .models import FoodImage
from .ml_models import ModelService
import os

class FoodClassifierViewSet(viewsets.ModelViewSet):
    model_service = ModelService()
    
    @action(detail=False, methods=['POST'])
    def classify_image(self, request):
        if 'image' not in request.FILES:
            return Response({'error': 'No image provided'}, status=400)
            
        image_file = request.FILES['image']
        food_image = FoodImage.objects.create(image=image_file)
        
        # Get predictions
        predictions = self.model_service.predict(food_image.image.path)
        food_image.predictions = predictions
        food_image.save()
        
        return Response({
            'id': food_image.id,
            'predictions': predictions,
            'image_url': food_image.image.url
        })

# food_classifier/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import FoodClassifierViewSet

router = DefaultRouter()
router.register(r'classifier', FoodClassifierViewSet, basename='classifier')

urlpatterns = [
    path('api/', include(router.urls)),
]

# training/augmentation.py
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AugmentationPipeline:
    def __init__(self):
        self.train_transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.ElasticTransform(p=0.3),
            ], p=0.2),
            A.OneOf([
                A.GaussNoise(p=0.3),
                A.ISONoise(p=0.3),
                A.MultiplicativeNoise(p=0.3),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.GaussianBlur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        self.val_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

# training/metrics.py
from sklearn.metrics import precision_recall_fscore_support
import torch
import numpy as np

class MetricsTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.sandwich_preds = []
        self.sandwich_labels = []
        self.ingredient_preds = []
        self.ingredient_labels = []
        self.dressing_preds = []
        self.dressing_labels = []
    
    def update(self, preds, labels):
        self.sandwich_preds.extend(preds['sandwich_type'].argmax(1).cpu().numpy())
        self.sandwich_labels.extend(labels['sandwich_type'].cpu().numpy())
        self.ingredient_preds.extend((preds['ingredients'] > 0.5).float().cpu().numpy())
        self.ingredient_labels.extend(labels['ingredients'].cpu().numpy())
        self.dressing_preds.extend(preds['dressing_rec'].argmax(1).cpu().numpy())
        self.dressing_labels.extend(labels['dressing_type'].cpu().numpy())
    
    def compute(self):
        sandwich_metrics = precision_recall_fscore_support(
            self.sandwich_labels, self.sandwich_preds, average='weighted'
        )
        
        ingredient_metrics = precision_recall_fscore_support(
            np.array(self.ingredient_labels).reshape(-1),
            np.array(self.ingredient_preds).reshape(-1),
            average='weighted'
        )
        
        dressing_metrics = precision_recall_fscore_support(
            self.dressing_labels, self.dressing_preds, average='weighted'
        )
        
        return {
            'sandwich': {
                'precision': sandwich_metrics[0],
                'recall': sandwich_metrics[1],
                'f1': sandwich_metrics[2]
            },
            'ingredients': {
                'precision': ingredient_metrics[0],
                'recall': ingredient_metrics[1],
                'f1': ingredient_metrics[2]
            },
            'dressing': {
                'precision': dressing_metrics[0],
                'recall': dressing_metrics[1],
                'f1': dressing_metrics[2]
            }
        }

# training/rl_component.py
class RLComponent:
    def __init__(self, model, buffer_size=10000):
        self.model = model
        self.replay_buffer = []
        self.buffer_size = buffer_size
    
    def add_experience(self, state, action, reward, next_state):
        if len(self.replay_buffer) >= self.buffer_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append((state, action, reward, next_state))
    
    def update_model(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return
        
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        
        # Convert to tensors
        states = torch.stack(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        next_states = torch.stack(next_states)
        
        # Compute TD error
        current_q = self.model(states)
        next_q = self.model(next_states).detach()
        target_q = rewards + 0.99 * next_q.max(1)[0]
        
        # Update model
        loss = nn.MSELoss()(current_q.gather(1, actions.unsqueeze(1)), target_q.unsqueeze(1))
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()

# training/loss_functions.py
class CombinedLoss(nn.Module):
    def __init__(self, weights={'sandwich': 1.0, 'ingredients': 1.0, 'dressing': 1.0}):
        super().__init__()
        self.weights = weights
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = FocalLoss()
    
    def forward(self, predictions, targets):
        sandwich_loss = self.focal_loss(predictions['sandwich_type'], targets['sandwich_type'])
        ingredient_loss = self.bce_loss(predictions['ingredients'], targets['ingredients'])
        dressing_loss = self.ce_loss(predictions['dressing_rec'], targets['dressing_type'])
        
        return (
            self.weights['sandwich'] * sandwich_loss +
            self.weights['ingredients'] * ingredient_loss +
            self.weights['dressing'] * dressing_loss
        )

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

# airflow/dags/training_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def train_and_evaluate(**context):
    # Set up training
    model = UnifiedFoodClassifier()
    augmentation = AugmentationPipeline()
    metrics_tracker = MetricsTracker()
    rl_component = RLComponent(model)
    
    # MLflow logging
    mlflow.set_experiment("food-classification")
    with mlflow.start_run():
        # Training loop
        for epoch in range(num_epochs):
            train_metrics = train_epoch(
                model, train_loader, augmentation.train_transform,
                metrics_tracker, rl_component
            )
            
            val_metrics = validate_epoch(
                model, val_loader, augmentation.val_transform,
                metrics_tracker
            )
            
            # Log metrics
            mlflow.log_metrics({
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                **{f'val_{k}_{m}': v 
                   for k, metrics in val_metrics['metrics'].items()
                   for m, v in metrics.items()}
            })
        
        # Save model
        mlflow.pytorch.log_model(model, "model")

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'food_classification_training',
    default_args=default_args,
    description='Food classification training pipeline',
    schedule_interval=timedelta(days=1)
)

train_task = PythonOperator(
    task_id='train_and_evaluate',
    python_callable=train_and_evaluate,
    dag=dag
)

Complete Food Classification System Implementation

# food_classifier/models.py
from django.db import models
import json
from datetime import datetime

class FoodImage(models.Model):
    image = models.ImageField(upload_to='food_images/%Y/%m/%d/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    predictions = models.JSONField(null=True)
    quality_metrics = models.JSONField(null=True)
    model_version = models.CharField(max_length=50)
    ab_test_group = models.CharField(max_length=20)
    
    def get_predictions(self):
        return json.loads(self.predictions) if self.predictions else {}
    
    def get_quality_metrics(self):
        return json.loads(self.quality_metrics) if self.quality_metrics else {}

# food_classifier/services/image_quality.py
import cv2
import numpy as np
from scipy.ndimage import laplace
from typing import Dict, Tuple

class ImageQualityService:
    def __init__(self, config: Dict):
        self.min_blur_score = config['quality_thresholds']['min_blur_score']
        self.min_exposure_score = config['quality_thresholds']['min_exposure_score']
        self.min_resolution = tuple(config['quality_thresholds']['min_resolution'])

    def assess_image(self, image_path: str) -> Dict:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        blur_score = self._calculate_blur_score(gray)
        exposure_score = self._analyze_exposure(gray)
        resolution_score = self._check_resolution(image.shape)
        
        return {
            'blur_score': blur_score,
            'exposure_score': exposure_score,
            'resolution_score': resolution_score,
            'is_acceptable': (
                blur_score > self.min_blur_score and 
                exposure_score > self.min_exposure_score and
                resolution_score > 0.8
            )
        }
    
    def _calculate_blur_score(self, gray_image: np.ndarray) -> float:
        laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        return min(1.0, laplacian_var / 500.0)  # Normalize to [0,1]
    
    def _analyze_exposure(self, gray_image: np.ndarray) -> float:
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # Normalize
        
        # Calculate entropy as a measure of exposure quality
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        max_entropy = np.log2(256)
        
        return entropy / max_entropy
    
    def _check_resolution(self, shape: Tuple[int, int, int]) -> float:
        height, width = shape[:2]
        min_height, min_width = self.min_resolution
        
        return min(
            1.0,
            min(height / min_height, width / min_width)
        )

# food_classifier/services/model_service.py
import torch
import torchvision.transforms as transforms
from PIL import Image
import mlflow
from typing import Dict
from .image_quality import ImageQualityService
from ..ml_models.unified_food_classifier import UnifiedFoodClassifier, FOOD_CATEGORIES

class ModelService:
    def __init__(self, config: Dict):
        self.model_a = None
        self.model_b = None
        self.quality_service = ImageQualityService(config)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.load_models()
    
    def load_models(self):
        mlflow.set_tracking_uri("http://localhost:5000")
        
        # Load primary model (A)
        latest_run_a = mlflow.search_runs(
            experiment_ids=['1'],
            filter_string="tags.production='true'",
            order_by=['start_time DESC']
        ).iloc[0]
        self.model_a = mlflow.pytorch.load_model(
            latest_run_a.artifact_uri + "/model"
        )
        self.model_a.eval()
        
        # Load challenger model (B) if available
        try:
            latest_run_b = mlflow.search_runs(
                experiment_ids=['1'],
                filter_string="tags.ab_test='true'",
                order_by=['start_time DESC']
            ).iloc[0]
            self.model_b = mlflow.pytorch.load_model(
                latest_run_b.artifact_uri + "/model"
            )
            self.model_b.eval()
        except IndexError:
            self.model_b = None
    
    def predict(self, image_path: str, ab_test_group: str = 'A') -> Dict:
        # Assess image quality
        quality_metrics = self.quality_service.assess_image(image_path)
        
        if not quality_metrics['is_acceptable']:
            return {
                'error': 'Image quality below threshold',
                'quality_metrics': quality_metrics
            }
        
        # Process image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        
        # Select model based on A/B test group
        model = self.model_a if ab_test_group == 'A' else self.model_b
        if model is None:
            model = self.model_a
        
        # Generate predictions
        with torch.no_grad():
            predictions = model(image_tensor)
        
        processed_predictions = self.process_predictions(predictions)
        
        return {
            'predictions': processed_predictions,
            'quality_metrics': quality_metrics,
            'model_version': ab_test_group
        }
    
    def process_predictions(self, predictions: Dict) -> Dict:
        sandwich_probs = torch.softmax(predictions['sandwich_type'][0], dim=0)
        ingredient_probs = torch.sigmoid(predictions['ingredients'][0])
        dressing_probs = torch.softmax(predictions['dressing_rec'][0], dim=0)
        
        return {
            'sandwich_type': {
                k: float(sandwich_probs[v]) 
                for k, v in FOOD_CATEGORIES['SANDWICH_TYPES'].items()
            },
            'ingredients': {
                k: float(ingredient_probs[v])
                for k, v in FOOD_CATEGORIES['SALAD_INGREDIENTS'].items()
                if float(ingredient_probs[v]) > 0.5
            },
            'recommended_dressings': {
                k: float(dressing_probs[v])
                for k, v in FOOD_CATEGORIES['DRESSING_TYPES'].items()
            }
        }

# food_classifier/services/drift_detector.py
from typing import Dict, List
import numpy as np
from scipy import stats
import pandas as pd
from datetime import datetime, timedelta

class DriftDetector:
    def __init__(self, config: Dict):
        self.window_size = pd.Timedelta(config['drift_detection']['window_size'])
        self.threshold = config['drift_detection']['threshold']
        self.min_samples = 1000
    
    def detect_drift(self, recent_predictions: List[Dict], 
                    historical_predictions: List[Dict]) -> Dict:
        if len(recent_predictions) < self.min_samples:
            return {'drift_detected': False, 'reason': 'Insufficient samples'}
        
        recent_df = pd.DataFrame(recent_predictions)
        historical_df = pd.DataFrame(historical_predictions)
        
        drift_metrics = {
            'distribution_shift': self._test_distribution_shift(
                recent_df, historical_df
            ),
            'accuracy_decline': self._test_accuracy_decline(
                recent_df, historical_df
            ),
            'confidence_shift': self._test_confidence_shift(
                recent_df, historical_df
            )
        }
        
        drift_detected = any(
            metric > self.threshold for metric in drift_metrics.values()
        )
        
        return {
            'drift_detected': drift_detected,
            'metrics': drift_metrics
        }
    
    def _test_distribution_shift(self, recent_df: pd.DataFrame, 
                               historical_df: pd.DataFrame) -> float:
        # Kolmogorov-Smirnov test for distribution shift
        _, p_value = stats.ks_2samp(
            recent_df['confidence'].values,
            historical_df['confidence'].values
        )
        return 1 - p_value
    
    def _test_accuracy_decline(self, recent_df: pd.DataFrame,
                             historical_df: pd.DataFrame) -> float:
        recent_accuracy = (
            recent_df['predicted_label'] == recent_df['true_label']
        ).mean()
        historical_accuracy = (
            historical_df['predicted_label'] == historical_df['true_label']
        ).mean()
        
        return max(0, historical_accuracy - recent_accuracy)
    
    def _test_confidence_shift(self, recent_df: pd.DataFrame,
                             historical_df: pd.DataFrame) -> float:
        # Test for significant shift in prediction confidence
        _, p_value = stats.ttest_ind(
            recent_df['confidence'].values,
            historical_df['confidence'].values
        )
        return 1 - p_value

# training/ab_testing.py
import numpy as np
from typing import Dict, List
from scipy import stats

class ABTestingFramework:
    def __init__(self, config: Dict):
        self.traffic_split = config['ab_testing']['traffic_split']
        self.min_sample_size = config['ab_testing']['min_sample_size']
        self.significance_level = config['ab_testing']['significance_level']
    
    def assign_group(self) -> str:
        return 'B' if np.random.random() < self.traffic_split[1] else 'A'
    
    def evaluate_test(self, group_a_results: List[Dict],
                     group_b_results: List[Dict]) -> Dict:
        if len(group_a_results) < self.min_sample_size or \
           len(group_b_results) < self.min_sample_size:
            return {
                'status': 'insufficient_data',
                'recommendation': None
            }
        
        # Convert to numpy arrays for statistical testing
        a_metrics = self._calculate_metrics(group_a_results)
        b_metrics = self._calculate_metrics(group_b_results)
        
        # Perform statistical tests
        accuracy_significant = self._test_significance(
            a_metrics['accuracy'],
            b_metrics['accuracy']
        )
        latency_significant = self._test_significance(
            a_metrics['latency'],
            b_metrics['latency']
        )
        
        # Make recommendation
        if accuracy_significant and b_metrics['accuracy'] > a_metrics['accuracy']:
            recommendation = 'promote_b'
        elif accuracy_significant and b_metrics['accuracy'] < a_metrics['accuracy']:
            recommendation = 'terminate_test'
        else:
            recommendation = 'continue_test'
        
        return {
            'status': 'active',
            'recommendation': recommendation,
            'metrics': {
                'group_a': a_metrics,
                'group_b': b_metrics,
                'significance': {
                    'accuracy': accuracy_significant,
                    'latency': latency_significant
                }
            }
        }
    
    def _calculate_metrics(self, results: List[Dict]) -> Dict:
        accuracy = np.mean([r['correct'] for r in results])
        latency = np.mean([r['latency'] for r in results])
        return {'accuracy': accuracy, 'latency': latency}
    
    def _test_significance(self, a_values: np.ndarray,
                          b_values: np.ndarray) -> bool:
        _, p_value = stats.ttest_ind(a_values, b_values)
        return p_value < self.significance_level

# airflow/dags/training_pipeline.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from typing import Dict

def prepare_training_data(**context) -> Dict:
    # Load and preprocess training data
    return {
        'training_data_path': '/path/to/processed/data',
        'metadata': {
            'num_samples': 10000,
            'categories': ['sandwich', 'salad']
        }
    }

def train_model(**context) -> Dict:
    ti = context['task_instance']
    training_data = ti.xcom_pull(task_ids='prepare_training_data')
    
    # Train model using preprocessed data
    return {
        'model_path': '/path/to/trained/model',
        'metrics': {
            'accuracy': 0.92,
            'loss': 0.08
        }
    }

def evaluate_model(**context) -> Dict:
    ti = context['task_instance']
    model_info = ti.xcom_pull(task_ids='train_model')
    
    # Evaluate model performance
    return {
        'evaluation_results': {
            'accuracy': 0.90,
            'precision': 0.89,
            'recall': 0.91
        }
    }

def check_drift(**context) -> Dict:
    # Check for model drift
    return {
        'drift_detected': False,
        'drift_metrics': {
            'distribution_shift': 0.02,
            'accuracy_decline': 0.01
        }
    }

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'food_classification_pipeline',
    default_args=default_args,
    description='Food classification training pipeline',
    schedule_interval=timedelta(days=1)
)

prepare_data = PythonOperator(
    task_id='prepare_training_data',
    python_callable=prepare_training_data,
    dag=dag
)

train = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

evaluate = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag
)

drift_check = PythonOperator(
    task_id='check_drift',
    python_callable=check_drift,
    dag=dag
)

# Set up task dependencies
prepare_data >> train >> evaluate >> drift_check

