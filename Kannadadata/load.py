from datasets import load_dataset

ds = load_dataset(
    "SPRINGLab/IndicTTS_Kannada",
    cache_dir="E:/ProJect/Civic Complaint/Kannadadata"
)

print(ds)