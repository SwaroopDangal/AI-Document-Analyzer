from huggingface_hub import snapshot_download

print("Starting model download...")
print("Model: sshleifer/distilbart-cnn-6-6")
print()

path = snapshot_download(
    repo_id="sshleifer/distilbart-cnn-6-6",
    local_dir="./models/distilbart-cnn-6-6"
)

print()
print("===================================")
print("Download completed!")
print("Model location:")
print(path)
print("===================================")