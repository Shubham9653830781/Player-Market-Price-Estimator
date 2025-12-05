
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import TargetEncoder

def build_preprocessor(num_features, cat_features, high_card_cats=None):
    num_pipe = StandardScaler()
    cat_pipe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    transformers = [('num', num_pipe, num_features)]
    if high_card_cats:
        te = TargetEncoder()
        transformers.append(('te', te, high_card_cats))
    transformers.append(('cat', cat_pipe, [c for c in cat_features if c not in (high_card_cats or [])]))
    preprocessor = ColumnTransformer(transformers, remainder='drop')
    return preprocessor
