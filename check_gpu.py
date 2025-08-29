import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
try:
    print("cuda version:", torch.version.cuda)
except Exception:
    pass
try:
    print("device count:", torch.cuda.device_count())
    if torch.cuda.device_count() > 0:
        print("device name:", torch.cuda.get_device_name(0))
except Exception as e:
    print("cuda device query error:", e)
