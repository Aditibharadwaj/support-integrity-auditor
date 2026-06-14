from huggingface_hub import HfApi
api = HfApi()
api.add_space_secret(
    repo_id="aditibharadwaj/support-integrity-auditor",  # your actual Space name
    key="MISTRAL_API_KEY",
    value="OT13NNHFIORFqDcajjtbysA3PAMptO9W",
)