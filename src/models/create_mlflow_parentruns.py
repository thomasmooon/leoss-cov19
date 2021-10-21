import mlflow

# cox
models = ["cox", "deepsurv", "xgb", "weiAft","rsf"] 
ENDPOINT = "death"
for MODEL in models:
    run_name = f"leoss_{MODEL}_{ENDPOINT}_nCV"
    with mlflow.start_run(run_name = run_name) as run:
        mlflow.set_tags({"ENDPOINT":ENDPOINT,"MODEL":MODEL})
        with mlflow.start_run(run_name = "dummy", nested=True) as run:
            pass
