```py
from huggingface_hub import snapshot_download

snapshot_download(
  repo_id="meta-llama/Llama-3.2-3B-Instruct",
  local_dir="AIModel/llama" #saved in AIModel folder's llama subfolder
)
```

### login in huggingface will do the download a bit faster.
```py
hf auth login -- token get_login_token_from_account
python DownloadModel.py
```
