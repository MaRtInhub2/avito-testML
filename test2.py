import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

TRAIN_PATH = "train_df.pq" # обучающая выборка
TEST_PATH = "test_df.pq" # тестовая выборка
TARGET = "item_contact" # целевая переменная (0/1)
ID_COLS = ["query_id", "item_id"]
RANDOM_STATE = 42

print("Loading data...")

train = pd.read_parquet(TRAIN_PATH)
test = pd.read_parquet(TEST_PATH)

y = train[TARGET]
X = train.drop(columns=[TARGET])

# Сохраняем id из test, чтобы потом собрать solution.csv
test_ids = test[ID_COLS].copy()
cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

# Использую stratify, чтобы сохранить баланс классов
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# Обучение модели на train с контролем качества на validation
model = CatBoostClassifier(
    iterations=500,
    depth=8,
    learning_rate=0.1,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=RANDOM_STATE,
    verbose=100,
    cat_features=cat_features,
    allow_writing_files=False,
    thread_count=-1
)

print("Training...")
model.fit(
    X_train, y_train,
    eval_set=(X_valid, y_valid), # валидация для early stopping
    use_best_model=True # сохраняю лучшую итерацию
)


valid_pred = model.predict_proba(X_valid)[:, 1]
roc = roc_auc_score(y_valid, valid_pred)
print(f"Validation ROC-AUC: {roc:.5f}")

# Финальное обучение на всей обучающей выборке
best_iters = model.get_best_iteration()

final_model = CatBoostClassifier(
    iterations=best_iters,
    depth=8,
    learning_rate=0.1,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=RANDOM_STATE,
    verbose=100,
    cat_features=cat_features,
    allow_writing_files=False,
    thread_count=-1
)

print("Training on full data...")
final_model.fit(X, y)

# Предсказания для тестовой выборки
test_pred = final_model.predict_proba(test)[:, 1]

solution = test_ids.copy()
solution["item_contact"] = test_pred
solution.to_csv("solution1.csv", index=False)
print("Saved solution1.csv")
