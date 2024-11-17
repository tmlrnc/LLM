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

