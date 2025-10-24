"""
Utility functions for Medical NER and Feature Extraction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def extract_entities_from_text(text, nlp_model):
    """
    Extract named entities from text using the provided NER model.
    
    Args:
        text: Input text string
        nlp_model: spaCy NER model
        
    Returns:
        dict: Dictionary with entity counts by type and list of entity texts
    """
    if pd.isna(text) or str(text).strip() == '' or str(text) == '___':
        return {
            'entity_count': 0,
            'disease_count': 0,
            'chemical_count': 0,
            'entities': []
        }
    
    try:
        doc = nlp_model(str(text)[:10000])  # Limit text length for efficiency
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        disease_count = sum(1 for _, label in entities if label == 'DISEASE')
        chemical_count = sum(1 for _, label in entities if label == 'CHEMICAL')
        
        return {
            'entity_count': len(entities),
            'disease_count': disease_count,
            'chemical_count': chemical_count,
            'entities': entities
        }
    except Exception as e:
        return {
            'entity_count': 0,
            'disease_count': 0,
            'chemical_count': 0,
            'entities': []
        }


def extract_entities_transformer(text, pipeline, max_length=512):
    """
    Extract entities using Hugging Face transformer pipeline.
    
    Args:
        text: Input text string
        pipeline: Hugging Face NER pipeline
        max_length: Maximum sequence length
        
    Returns:
        dict: Dictionary with entities and counts
    """
    if pd.isna(text) or str(text).strip() == '':
        return {
            'entity_count': 0,
            'entities': []
        }
    
    try:
        entities = pipeline(str(text)[:max_length])
        
        return {
            'entity_count': len(entities),
            'entities': [(ent['word'], ent['entity_group'], ent['score']) 
                        for ent in entities]
        }
    except Exception as e:
        return {
            'entity_count': 0,
            'entities': []
        }


def combine_text_fields(row: pd.Series, fields: List[str]) -> str:
    """
    Combine multiple text fields into a single string.
    
    Args:
        row: DataFrame row
        fields: List of column names to combine
        
    Returns:
        str: Combined text
    """
    texts = []
    for field in fields:
        if field in row.index and pd.notna(row[field]):
            texts.append(str(row[field]))
    return ' '.join(texts).strip()


def batch_process_ner(data: pd.DataFrame, 
                      nlp_model,
                      text_column: str,
                      batch_size: int = 1000,
                      save_checkpoint: bool = True,
                      checkpoint_path: Optional[str] = None) -> List[Dict]:
    """
    Process NER on dataset in batches to manage memory.
    
    Args:
        data: Input DataFrame
        nlp_model: spaCy NER model
        text_column: Column name containing text
        batch_size: Number of records per batch
        save_checkpoint: Whether to save intermediate results
        checkpoint_path: Path to save checkpoints
        
    Returns:
        List of NER results
    """
    num_batches = len(data) // batch_size + 1
    all_results = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data))
        
        print(f"Processing batch {i+1}/{num_batches} "
              f"(records {start_idx} to {end_idx})...")
        
        batch = data.iloc[start_idx:end_idx]
        batch_results = batch[text_column].apply(
            lambda x: extract_entities_from_text(x, nlp_model)
        ).tolist()
        
        all_results.extend(batch_results)
        
        # Save checkpoint
        if save_checkpoint and checkpoint_path and i % 10 == 0:
            print(f"Saving checkpoint at batch {i}...")
            checkpoint_df = pd.DataFrame(all_results[:len(batch_results)])
            checkpoint_df.to_pickle(f"{checkpoint_path}_batch_{i}.pkl")
    
    return all_results


def aggregate_by_admission(data: pd.DataFrame,
                          ner_features: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Aggregate records by hospital admission (hadm_id).
    
    Args:
        data: Input DataFrame with individual records
        ner_features: List of NER feature columns to aggregate
        
    Returns:
        DataFrame aggregated by admission
    """
    if ner_features is None:
        ner_features = ['entity_count', 'disease_count', 'chemical_count']
    
    agg_dict = {
        # Demographics (take first value)
        'subject_id': 'first',
        'admission_type': 'first',
        'admission_location': 'first',
        'discharge_location': 'first',
        'insurance': 'first',
        'marital_status': 'first',
        'race': 'first',
        'gender': 'first',
        'anchor_age': 'first',
        
        # Target variable
        'description': 'first',
        'drg_type': 'first',
        'drg_severity': 'first',
        'drg_mortality': 'first',
        
        # Count of records per admission
        'hadm_id': 'count'
    }
    
    # Add NER features (sum)
    for feat in ner_features:
        if feat in data.columns:
            agg_dict[feat] = 'sum'
    
    agg_data = data.groupby('hadm_id').agg(agg_dict)
    agg_data = agg_data.rename(columns={'hadm_id': 'record_count'})
    
    return agg_data


def prepare_features(data: pd.DataFrame,
                    categorical_cols: Optional[List[str]] = None,
                    numerical_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Prepare features for machine learning models.
    
    Args:
        data: Input DataFrame
        categorical_cols: List of categorical columns to encode
        numerical_cols: List of numerical columns
        
    Returns:
        Tuple of (feature DataFrame, label encoders dict)
    """
    from sklearn.preprocessing import LabelEncoder
    
    if categorical_cols is None:
        categorical_cols = ['admission_type', 'admission_location', 
                          'discharge_location', 'insurance', 'marital_status',
                          'race', 'gender', 'drg_type']
    
    if numerical_cols is None:
        numerical_cols = ['anchor_age', 'record_count', 'entity_count',
                        'disease_count', 'chemical_count', 
                        'drg_severity', 'drg_mortality']
    
    feature_data = data.copy()
    label_encoders = {}
    
    # Encode categorical variables
    for col in categorical_cols:
        if col in feature_data.columns:
            le = LabelEncoder()
            feature_data[f'{col}_encoded'] = le.fit_transform(
                feature_data[col].fillna('Unknown')
            )
            label_encoders[col] = le
    
    # Prepare feature list
    encoded_features = [f'{col}_encoded' for col in categorical_cols 
                       if col in feature_data.columns]
    all_features = numerical_cols + encoded_features
    
    # Select features
    X = feature_data[all_features].fillna(0)
    
    return X, label_encoders


def get_entity_vocabulary(ner_results: List[Dict], 
                         min_count: int = 5) -> Dict[str, int]:
    """
    Build vocabulary of entities from NER results.
    
    Args:
        ner_results: List of NER result dictionaries
        min_count: Minimum occurrence count for inclusion
        
    Returns:
        Dictionary mapping entity text to count
    """
    from collections import Counter
    
    all_entities = []
    for result in ner_results:
        all_entities.extend([ent[0].lower() for ent in result.get('entities', [])])
    
    entity_counts = Counter(all_entities)
    
    # Filter by minimum count
    vocab = {entity: count for entity, count in entity_counts.items()
            if count >= min_count}
    
    return vocab


def create_entity_features(data: pd.DataFrame,
                          entity_column: str,
                          vocab: Dict[str, int],
                          top_k: int = 100) -> pd.DataFrame:
    """
    Create binary features for presence of top entities.
    
    Args:
        data: Input DataFrame
        entity_column: Column containing entity lists
        vocab: Entity vocabulary with counts
        top_k: Number of top entities to use as features
        
    Returns:
        DataFrame with binary entity features
    """
    # Get top entities
    top_entities = sorted(vocab.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_entity_names = [ent[0] for ent in top_entities]
    
    # Create binary features
    entity_features = pd.DataFrame(index=data.index)
    
    for entity in top_entity_names:
        col_name = f'has_{entity.replace(" ", "_")[:30]}'
        entity_features[col_name] = data[entity_column].apply(
            lambda entities: 1 if any(ent[0].lower() == entity 
                                     for ent in entities) else 0
        )
    
    return entity_features


def evaluate_model(y_true, y_pred, model_name: str = "Model") -> Dict:
    """
    Evaluate classification model performance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        
    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.metrics import (accuracy_score, precision_score, 
                                 recall_score, f1_score, classification_report)
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_weighted': precision_score(y_true, y_pred, 
                                             average='weighted', 
                                             zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, 
                                       average='weighted',
                                       zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, 
                               average='weighted',
                               zero_division=0),
        'classification_report': classification_report(y_true, y_pred,
                                                      zero_division=0)
    }
    
    return results


def save_processed_data(data: pd.DataFrame, 
                       filepath: str,
                       format: str = 'pickle'):
    """
    Save processed data to file.
    
    Args:
        data: DataFrame to save
        filepath: Output file path
        format: File format ('pickle', 'csv', 'parquet')
    """
    if format == 'pickle':
        data.to_pickle(filepath)
    elif format == 'csv':
        data.to_csv(filepath, index=False)
    elif format == 'parquet':
        data.to_parquet(filepath, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Data saved to {filepath} ({format} format)")


def load_processed_data(filepath: str, format: str = 'pickle') -> pd.DataFrame:
    """
    Load processed data from file.
    
    Args:
        filepath: Input file path
        format: File format ('pickle', 'csv', 'parquet')
        
    Returns:
        DataFrame
    """
    if format == 'pickle':
        data = pd.read_pickle(filepath)
    elif format == 'csv':
        data = pd.read_csv(filepath)
    elif format == 'parquet':
        data = pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Data loaded from {filepath} ({format} format)")
    return data

