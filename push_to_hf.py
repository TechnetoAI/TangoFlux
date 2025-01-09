from huggingface_hub import HfApi
api = HfApi()

api.upload_file(
    path_or_fileobj="/home/sang/work/TangoFlux/outputs/best/scheduler.bin",
    path_in_repo="scheduler.bin",
    repo_id="techneto/tango-music-cp28",
)

api.upload_file(
    path_or_fileobj="/home/sang/work/TangoFlux/outputs/best/random_states_0.pkl",
    path_in_repo="random_states_0.pkl",
    repo_id="techneto/tango-music-cp28",
)

api.upload_file(
    path_or_fileobj="/home/sang/work/TangoFlux/outputs/best/optimizer.bin",
    path_in_repo="optimizer.bin",
    repo_id="techneto/tango-music-cp28",
)

api.upload_file(
    path_or_fileobj="/home/sang/work/TangoFlux/outputs/best/model_1.safetensors",
    path_in_repo="model_1.safetensors",
    repo_id="techneto/tango-music-cp28",
)

api.upload_file(
    path_or_fileobj="/home/sang/work/TangoFlux/outputs/best/model.safetensors",
    path_in_repo="model.safetensors",
    repo_id="techneto/tango-music-cp28",
)
