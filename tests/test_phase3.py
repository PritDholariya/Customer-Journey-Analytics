import pytest
import pandas as pd
from cja.config import USER_FEATURES_FILE, CLEAN_EVENTS_FILE

def test_feature_table_shape():
    if not USER_FEATURES_FILE.exists() or not CLEAN_EVENTS_FILE.exists():
        pytest.skip("Data files not generated.")
        
    clean_df = pd.read_parquet(CLEAN_EVENTS_FILE)
    feat_df = pd.read_parquet(USER_FEATURES_FILE)
    
    # Ensure there is exactly one row per unique user
    assert len(feat_df) == clean_df['user_id'].nunique(), "Row count does not match unique users."
    
def test_no_nans():
    if not USER_FEATURES_FILE.exists():
        pytest.skip("Feature file not generated.")
        
    feat_df = pd.read_parquet(USER_FEATURES_FILE)
    
    # Ensure division by zero or aggregation didn't leak NaNs
    assert feat_df.isna().sum().sum() == 0, "Found NaNs in the feature table."

def test_funnel_logic():
    if not USER_FEATURES_FILE.exists():
        pytest.skip("Feature file not generated.")
        
    feat_df = pd.read_parquet(USER_FEATURES_FILE)
    
    # If converted is True, max_funnel_stage must be 2
    converted_users = feat_df[feat_df['converted'] == 1]
    assert (converted_users['max_funnel_stage'] == 2).all()
    
    # Ratios shouldn't exceed logical bounds (though realistically they can be > 1 
    # if someone adds to cart multiple times without viewing again, but let's just test they are >= 0)
    assert (feat_df['cart_to_view_ratio'] >= 0).all()
    assert (feat_df['purchase_to_cart_ratio'] >= 0).all()

def test_log_transforms():
    if not USER_FEATURES_FILE.exists():
        pytest.skip("Feature file not generated.")
        
    feat_df = pd.read_parquet(USER_FEATURES_FILE)
    
    # Ensure log1p applied correctly: log(1+0) = 0
    zero_views = feat_df[feat_df['n_views'] == 0]
    if not zero_views.empty:
        assert (zero_views['n_views_log'] == 0).all()
