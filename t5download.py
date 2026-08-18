from huggingface_hub import snapshot_download


print("Starting model download...")
print("Model: valhalla/t5-small-qg-hl")
print()


path = snapshot_download(
    repo_id="valhalla/t5-small-qg-hl",
    local_dir="./models/t5-small-qg-hl"
)


print()
print("===================================")
print("Download completed!")
print("Model location:")
print(path)
print("===================================")